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


np.random.seed(1101)

dtype_y = 'uint8'
model_fn = 'lemonhope.pkl'

img_size = 94
max_offset = 1
window = (img_size-2*max_offset, img_size-2*max_offset)

train_x, train_y, train_names = load_npys(which_set='train', image_size=img_size, resizing_method='shrink')
valid_x, valid_y, valid_names = load_npys(which_set='valid', image_size=img_size, resizing_method='shrink')

train_x = train_x.reshape((train_x.shape[0], 1, np.sqrt(train_x.shape[1]), np.sqrt(train_x.shape[1])))
valid_x = valid_x.reshape((valid_x.shape[0], 1, np.sqrt(valid_x.shape[1]), np.sqrt(valid_x.shape[1])))
train_x = np.cast[floatX](1 - train_x/255.)
valid_x = np.cast[floatX](1 - valid_x/255.)

# cropped_size = 40
# cropped_offset = 1
# cropped_window = (cropped_size-2*cropped_offset, cropped_size-2*cropped_offset)

# cropped_train_x, _, _ = load_npys(which_set='train', image_size=cropped_size, resizing_method='crop')
# cropped_valid_x, _, _ = load_npys(which_set='valid', image_size=cropped_size, resizing_method='crop')
#
# cropped_train_x = cropped_train_x.reshape((cropped_train_x.shape[0], 1, np.sqrt(cropped_train_x.shape[1]), np.sqrt(cropped_train_x.shape[1])))
# cropped_valid_x = cropped_valid_x.reshape((cropped_valid_x.shape[0], 1, np.sqrt(cropped_valid_x.shape[1]), np.sqrt(cropped_valid_x.shape[1])))
# cropped_train_x = np.cast[floatX](1 - cropped_train_x/255.)
# cropped_valid_x = np.cast[floatX](1 - cropped_valid_x/255.)

# csv_location = '/home/yoptar/git/subway-plankton/train_img_props.csv'
# pan_props = pd.read_csv(csv_location, sep='\t')

# # train_props = None
# # for name in train_names:
# #     prop = np.cast[floatX](pan_props[pan_props['file_name'] == name].drop(['class', 'file_name'], axis=1).values)
# #     if train_props is None:
# #         train_props = prop
# #     else:
# #         train_props = np.vstack((train_props, prop))
# # print('prepared train_props')
# train_props = np.load('/plankton/train_props.npy')
#
# # valid_props = None
# # for name in valid_names:
# #     prop = np.cast[floatX](pan_props[pan_props['file_name'] == name].drop(['class', 'file_name'], axis=1).values)
# #     if valid_props is None:
# #         valid_props = prop
# #     else:
# #         valid_props = np.vstack((valid_props, prop))
# # print('prepared valid_props')
# valid_props = np.load('/plankton/valid_props.npy')

# means = train_props.mean(axis=0)
# stds = np.std(train_props, axis=0)
# train_props = (train_props - means) / stds
# valid_props = (valid_props - means) / stds

len_in = train_x.shape[1]
len_out = train_y.shape[1]

# train_x = train_x  # cropped_train_x)  # , train_props)
# valid_x = valid_x  # cropped_valid_x)  # , valid_props)


unique_labels = read_labels()
n_classes = len(unique_labels)
label_to_int = {unique_labels[i]: i for i in range(n_classes)}


# def heat_ys(y):
#     cl = zip(0.7**np.arange(1, 10), np.exp(np.arange(2, 11)))
#     heat_ys.iter = min(heat_ys.iter, len(cl)-1)
#     res = heated_targetings(label_to_int, y,
#                             cl[heat_ys.iter][0], cl[heat_ys.iter][0], cl[heat_ys.iter][1])
#     heat_ys.counter += 1
#     if heat_ys.counter % 3 == 0:
#         heat_ys.iter += 1
#     return res
# heat_ys.counter = 0
# heat_ys.iter = 0

cost = neg_log_likelihood
valid_cost = cost
train_y_modifier = identity

# print('merging train and valid')
# train_x = np.vstack((train_x, valid_x))
# train_y = np.vstack((train_y, valid_y))

# cost = soft_log_likelihood
# train_y_modifier = heat_ys

if os.path.isfile(model_fn) and True:
    logger.write('Loading model from %s...' % model_fn)
    mlp = cPickle.load(open(model_fn, 'rb'))
else:
    prelu_alpha = 0.25
    mlp = MLP([
                ConvolutionalLayer((3, 3), 16, leaky_relu_alpha=1),
                BatchNormalization(),
                PReLU(prelu_alpha),

                ConvolutionalLayer((3, 3), 64, leaky_relu_alpha=0.1),
                MaxPool((2, 2)),
                BatchNormalization(),
                PReLU(prelu_alpha),

                ConvolutionalLayer((3, 3), 80, leaky_relu_alpha=0.1),
                BatchNormalization(),
                PReLU(prelu_alpha),

                ConvolutionalLayer((3, 3), 96, leaky_relu_alpha=0.1),
                BatchNormalization(),
                PReLU(prelu_alpha),

                ConvolutionalLayer((3, 3), 112, leaky_relu_alpha=0.1),
                BatchNormalization(),
                PReLU(prelu_alpha),

                ConvolutionalLayer((3, 3), 128, leaky_relu_alpha=0.1),
                BatchNormalization(),
                MaxPool((2, 2)),
                PReLU(prelu_alpha),

                ConvolutionalLayer((3, 3), 200, leaky_relu_alpha=0.1),
                BatchNormalization(),
                NonLinearity(activation=LeakyReLU(0.1)),

                ConvolutionalLayer((3, 3), 220, leaky_relu_alpha=0.1),
                BatchNormalization(),
                NonLinearity(activation=LeakyReLU(0.1)),

                ConvolutionalLayer((3, 3), 240, leaky_relu_alpha=0.1),
                BatchNormalization(),
                NonLinearity(activation=LeakyReLU(0.1)),

                ConvolutionalLayer((3, 3), 260, leaky_relu_alpha=0.1),
                BatchNormalization(),
                NonLinearity(activation=LeakyReLU(0.1)),

                ConvolutionalLayer((3, 3), 280, leaky_relu_alpha=0.1),
                BatchNormalization(),
                NonLinearity(activation=LeakyReLU(0.1)),

                ConvolutionalLayer((3, 3), 300, leaky_relu_alpha=0.1),
                BatchNormalization(),
                NonLinearity(activation=LeakyReLU(0.1)),

                MaxPool((2, 2)),

                Flatten(),

                Dropout(0.5, 1),
                DenseLayer(2000, leaky_relu_alpha=0.1),
                BatchNormalization(),
                Maxout(5),

                Dropout(0.8, 1),
                DenseLayer(2500),
                BatchNormalization(),
                Maxout(5),

                Dropout(0.8),
                DenseLayer(len_out),
                BatchNormalization(),
                NonLinearity(activation=T.nnet.softmax)
            ], (1,) + window,
        logger=logger)


## TODO move to mlp.get_updates
l2 = 1e-5
learning_rate = 5e-4  # np.exp(-2)
momentum = 0.995
epoch_count = 1000
batch_size = 32
minibatch_count = (train_y.shape[0]-1) // batch_size + 1
learning_decay = 1  # 0.5 ** (1./(10 * minibatch_count))
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


def randomize(dataset):
    return randomize_dataset_bc01(dataset[0], **randomization_params),

tr = Trainer(mlp, batch_size, learning_rate, train_x, train_y, valid_X=valid_x, valid_y=valid_y, method=method,
             momentum=momentum, lr_decay=learning_decay, lr_min=lr_min, l2=l2, mm_decay=momentum_decay, mm_min=mm_min,
             train_augmentation=randomize, valid_augmentation=randomize, valid_aug_count=valid_rnd_count,
             train_y_augmentation=train_y_modifier,
             model_file_name=model_fn, save_freq=5, save_in_different_files=True, epoch_count=epoch_count,
             cost_f=cost, valid_cost_f=valid_cost)
with open('monitors_%s.csv' % model_fn, 'w', 0) as monitor_csv:
    logger.write('#\tt_nll\tv_nll\ttime\trclass\tlearning rate\tmomentum\tl2_error')  # header
    for res in tr():
        logger.write('{epoch}\t{train_nll:.5f}\t{test_nll:.5f}\t{epoch_time:.1f}s\t{valid_misclass:.2f}%\t{lr:.10f}\t'
                     '{momentum:.9f}\t{l2_error:.5f}'.format(**res))
        if res['epoch'] == 0:
            print('\t'.join(res), file=monitor_csv)
        print('\t'.join([str(v) for v in res.itervalues()]), file=monitor_csv)