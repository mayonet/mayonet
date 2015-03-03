#!/usr/bin/env python2
from __future__ import division, print_function
import os
from pprint import pprint
from dataset import read_labels
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
        print(message)
        prefix = "" if message == '\n' else time.strftime("%Y.%m.%d %H:%M:%S --- ")
        self.log.write(prefix + message + "\n")

    def close(self):
        self.log.close()

logger_name = time.strftime("%Y-%m-%dT%H-%M-%S.log")
logger = Logger(logger_name)


np.random.seed(1100)

dtype_y = 'uint8'
model_fn = 'last_model_50_full.pkl'

unique_labels = read_labels()
n_classes = len(unique_labels)
label_to_int = {unique_labels[i]: i for i in range(n_classes)}
siphonophore_columns = []
for lbl, idx in label_to_int.items():
    if lbl.startswith('siphonophore') or lbl.startswith('acantharia'):
        siphonophore_columns.append(idx)
siphonophore_columns = np.array(siphonophore_columns)

img_size = 64
max_offset = 1
window = (img_size-2*max_offset, img_size-2*max_offset)

train_x, train_y = load_npys(which_set='train', image_size=img_size)
valid_x, valid_y = load_npys(which_set='valid', image_size=img_size)

# train_y = train_y[:, siphonophore_columns]
# train_x = train_x[np.max(train_y, axis=1) == 1, :]
# train_y = train_y[np.max(train_y, axis=1) == 1, :]
# valid_y = valid_y[:, siphonophore_columns]
# valid_x = valid_x[np.max(valid_y, axis=1) == 1, :]
# valid_y = valid_y[np.max(valid_y, axis=1) == 1, :]

train_x = train_x.reshape((train_x.shape[0], 1, np.sqrt(train_x.shape[1]), np.sqrt(train_x.shape[1])))
valid_x = valid_x.reshape((valid_x.shape[0], 1, np.sqrt(valid_x.shape[1]), np.sqrt(valid_x.shape[1])))
train_x = np.cast[floatX](1 - train_x/255.)
valid_x = np.cast[floatX](1 - valid_x/255.)


len_in = train_x.shape[1]
len_out = train_y.shape[1]

if os.path.isfile(model_fn):
    logger.write('Loading model from %s...' % model_fn)
    mlp = cPickle.load(open(model_fn, 'rb'))
else:
    prelu_alpha = 0.25
    mlp = MLP([
        # GaussianDropout(0.1),
        ConvolutionalLayer((3, 3), 16, train_bias=True, pad=0),
        # BatchNormalization(),
        MaxPool((2, 2)),
        # PReLU(prelu_alpha),
        NonLinearity(),

        # GaussianDropout(0.5),

        # ConvolutionalLayer((3, 3), 32, train_bias=True, pad=1, leaky_relu_alpha=prelu_alpha, max_kernel_norm=1.9),
        # BatchNormalization(),
        # PReLU(prelu_alpha),

        ConvolutionalLayer((3, 3), 32, train_bias=True, pad=0),
        # BatchNormalization(),
        MaxPool((2, 2)),
        # PReLU(prelu_alpha),
        NonLinearity(),

        ConvolutionalLayer((3, 3), 64, train_bias=True, pad=0),
        # BatchNormalization(),
        MaxPool((2, 2)),
        # PReLU(prelu_alpha),
        NonLinearity(),
        #
        # ConvolutionalLayer((3, 3), 128, train_bias=True, pad=1, leaky_relu_alpha=prelu_alpha, max_kernel_norm=1.9),
        # BatchNormalization(),
        # MaxPool((2, 2), (2, 2)),
        # PReLU(prelu_alpha),

        Flatten(),

        # # GaussianDropout(0.5),
        # DenseLayer(1000, max_col_norm=1.9),
        # # BatchNormalization(),
        # NonLinearity(),
        # # PReLU(prelu_alpha),

        # GaussianDropout(1),
        DenseLayer(1000, max_col_norm=1.9365),
        # BatchNormalization(),
        NonLinearity(),
        # PReLU(prelu_alpha),

        # GaussianDropout(1),
        DenseLayer(len_out, max_col_norm=1.9365),
        # BatchNormalization(),
        NonLinearity(activation=T.nnet.softmax)
    ], (1,) + window)



## TODO move to mlp.get_updates
l2 = 0  # 1e-5
learning_rate = 1e-2  # np.exp(-2)
momentum = 0.9
epoch_count = 1000
batch_size = 64
minibatch_count = train_x.shape[0] // batch_size
learning_decay = 0.5 ** (1./(250 * minibatch_count))
momentum_decay = 0.5 ** (1./(1000 * minibatch_count))
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
    # 'median_radii': [0, 0, 0, 1],
    # 'mean_radii': [0, 0, 0, 0, 1, 1, 2],
    'flip': True
}
print(randomization_params, file=logger)


def randomize(dataset):
    return randomize_dataset_bc01(dataset, **randomization_params)

tr = Trainer(mlp, batch_size, learning_rate, train_x, train_y, valid_X=valid_x, valid_y=valid_y, method=method,
             momentum=momentum, lr_decay=learning_decay, lr_min=lr_min, l2=l2, mm_decay=momentum_decay, mm_min=mm_min,
             train_augmentation=randomize, valid_augmentation=randomize, valid_aug_count=valid_rnd_count,
             model_file_name=model_fn, save_freq=1, epoch_count=epoch_count)
for res in tr():
    logger.write('{epoch}\t{train_nll:.5f}\t{test_nll:.5f}\t{epoch_time:.1f}s\t{valid_misclass:.2f}%%\t{lr}\t'
                 '{momentum}'.format(**res))