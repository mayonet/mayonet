#!/usr/bin/env python2
from __future__ import division, print_function
import pandas as pd
import os
from pprint import pprint
from dataset import read_labels, iterate_train_data_names
from hierarchy import heated_targetings
from np_dataset import load_npys
import theano
# theano.config.compute_test_value = 'warn'
from rotator import randomize_dataset_bc01

# theano.config.exception_verbosity = 'high'
theano.config.floatX = 'float32'
theano.config.blas.ldflags = '-lblas -lgfortran'
floatX = theano.config.floatX
from ann import *


class Logger(object):
    def __init__(self, filename="Default.log"):
        filename = os.path.join('logs', filename)
        self.log = open(filename, "a", 0)

    def write(self, message):
        if message != '\n':
            print(message)
            self.log.write(time.strftime("%Y.%m.%d %H:%M:%S --- ") + message + "\n")

    def close(self):
        self.log.close()

logger_name = time.strftime("%Y-%m-%dT%H-%M-%S.log")
logger = Logger(logger_name)


np.random.seed(1190)

dtype_y = 'uint8'
model_fn = 'amazon_3channel.pkl'

img_size = 52
max_offset = 2
window = (img_size-2*max_offset, img_size-2*max_offset)

train_x, train_y, train_names = load_npys(which_set='train', image_size=img_size, resizing_method='bluntresize')
valid_x, valid_y, valid_names = load_npys(which_set='valid', image_size=img_size, resizing_method='bluntresize')

train_x = train_x.reshape((train_x.shape[0], 1, np.sqrt(train_x.shape[1]), np.sqrt(train_x.shape[1])))
valid_x = valid_x.reshape((valid_x.shape[0], 1, np.sqrt(valid_x.shape[1]), np.sqrt(valid_x.shape[1])))
train_x = np.cast[floatX](1 - train_x/255.)
valid_x = np.cast[floatX](1 - valid_x/255.)

len_in = train_x.shape[1]
len_out = train_y.shape[1]

train_x = train_x  # cropped_train_x)  # , train_props)
valid_x = valid_x  # cropped_valid_x)  # , valid_props)


unique_labels = read_labels()
n_classes = len(unique_labels)
label_to_int = {unique_labels[i]: i for i in range(n_classes)}


cost = neg_log_likelihood
valid_cost = cost
train_y_modifier = identity

# print('merging train and valid')
# train_x = np.vstack((train_x, valid_x))
# train_y = np.vstack((train_y, valid_y))

if os.path.isfile(model_fn) and True:
    logger.write('Loading model from %s...' % model_fn)
    mlp = cPickle.load(open(model_fn, 'rb'))
else:
    prelu_alpha = 0.25
    mlp = MLP([
        MLP([
            GaussianDropout(0.03),
            ConvolutionalLayer((5, 5), 32, pad=2, leaky_relu_alpha=1),
            MaxPool((3, 3), (2, 2)),
            NonLinearity(),

            # GaussianDropout(0.03),
            ConvolutionalLayer((3, 3), 128, pad=2),
            NonLinearity(),

            ConvolutionalLayer((3, 3), 192, pad=2),
            NonLinearity(),

            ConvolutionalLayer((3, 3), 256, pad=1),
            NonLinearity(),
            MaxPool((3, 3), (2, 2)),

            Flatten(),

        ], logger=logger),

        MLP([

            # Dropout(0.5),
            DenseLayer(512),
            NonLinearity(),

            Dropout(0.5),
            DenseLayer(512),
            NonLinearity(),

            Dropout(0.5),
            DenseLayer(len_out),
            NonLinearity(activation=T.nnet.softmax)
        ], logger=logger),
    ], (1,) + window,  # (1,) + cropped_window  # , train_props.shape[1:]
        logger=logger)


## TODO move to mlp.get_updates
l2 = 0  # 1e-4
learning_rate = 5e-4  # np.exp(-2)
momentum = 0.99
epoch_count = 1000
batch_size = 64
minibatch_count = (train_y.shape[0]-1) // batch_size + 1
learning_decay = 0.5 ** (1./(100 * minibatch_count))
momentum_decay = 1  # 0.5 ** (1./(1000 * minibatch_count))
lr_min = 1e-15
mm_min = 0.5
valid_rnd_count = 1

method = 'nesterov'

logger.write('batch=%d, l2=%f, method=%s\nlr=%f, lr_decay=%f,\nmomentum=%f, momentum_decay=%f' %
             (batch_size, l2, method, learning_rate, learning_decay, momentum, momentum_decay))

scales = [1.01**(p*abs(p)) for p in map(lambda x: x/2., range(-7, 11))]

randomization_params = {
    'window': window,
    'scales': scales,
    'angles': range(360),
    'x_offsets': range(max_offset+1),
    'y_offsets': range(max_offset+1),
    'flip': True
}

cropped_randomization_params = {
    'window': window,
    'scales': (1,),
    'angles': (0, 90, 180, 270),
    'x_offsets': range(max_offset+1),
    'y_offsets': range(max_offset+1),
    'flip': True
}
print(randomization_params, file=logger)


def randomize(dataset):
    return [randomize_dataset_bc01(dataset[0], **randomization_params)]
           # [randomize_dataset_bc01(dataset[1], **cropped_randomization_params)]

tr = Trainer(mlp, batch_size, learning_rate, train_x, train_y, valid_X=valid_x, valid_y=valid_y, method=method,
             momentum=momentum, lr_decay=learning_decay, lr_min=lr_min, l2=l2, mm_decay=momentum_decay, mm_min=mm_min,
             train_augmentation=randomize, valid_augmentation=randomize, valid_aug_count=valid_rnd_count,
             train_y_augmentation=train_y_modifier,
             model_file_name=model_fn, save_freq=1, save_in_different_files=True, epoch_count=epoch_count,
             cost_f=cost, valid_cost_f=valid_cost)
with open('monitors_%s.csv' % model_fn, 'w', 0) as monitor_csv:
    logger.write('#\tt_nll\tv_nll\ttime\trclass\tlearning rate\tmomentum\tl2_error')  # header
    for res in tr():
        logger.write('{epoch}\t{train_nll:.5f}\t{test_nll:.5f}\t{epoch_time:.1f}s\t{valid_misclass:.2f}%\t{lr:.10f}\t'
                     '{momentum:.9f}\t{l2_error:.5f}'.format(**res))
        if res['epoch'] == 0:
            print('\t'.join(res), file=monitor_csv)
        print('\t'.join([str(v) for v in res.itervalues()]), file=monitor_csv)