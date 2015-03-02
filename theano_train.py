#!/usr/bin/env python2
from __future__ import division, print_function
from itertools import repeat
import os
from pprint import pprint
import time
import cPickle
from concurrent.futures import ProcessPoolExecutor
from dataset import read_labels
from np_dataset import load_npys
import numpy as np
import theano
import theano.tensor as T
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
        self.log.write(time.strftime("%Y.%m.%d %H:%M:%S --- ") + message + "\n")

    def close(self):
        self.log.close()

logger_name = time.strftime("%Y-%m-%dT%H-%M-%S.log")
logger = Logger(logger_name)


np.random.seed(1100)

dtype_y = 'uint8'
model_fn = 'last_model_wnorm.pkl'

unique_labels = read_labels()
n_classes = len(unique_labels)
label_to_int = {unique_labels[i]: i for i in range(n_classes)}
siphonophore_columns = []
for lbl, idx in label_to_int.items():
    if lbl.startswith('siphonophore') or lbl.startswith('acantharia'):
        siphonophore_columns.append(idx)
siphonophore_columns = np.array(siphonophore_columns)

train_x, train_y = load_npys(which_set='train', image_size=98)
valid_x, valid_y = load_npys(which_set='valid', image_size=98)

train_y = train_y[:, siphonophore_columns]
train_x = train_x[np.max(train_y, axis=1) == 1, :]
train_y = train_y[np.max(train_y, axis=1) == 1, :]
valid_y = valid_y[:, siphonophore_columns]
valid_x = valid_x[np.max(valid_y, axis=1) == 1, :]
valid_y = valid_y[np.max(valid_y, axis=1) == 1, :]

train_x = train_x.reshape((train_x.shape[0], 1, np.sqrt(train_x.shape[1]), np.sqrt(train_x.shape[1])))
valid_x = valid_x.reshape((valid_x.shape[0], 1, np.sqrt(valid_x.shape[1]), np.sqrt(valid_x.shape[1])))
train_x = np.cast[floatX](1 - train_x/255.)
valid_x = np.cast[floatX](1 - valid_x/255.)


len_in = train_x.shape[1]
len_out = train_y.shape[1]

# if os.path.isfile(model_fn):
#     logger.write('Loading model from %s...' % model_fn)
#     mlp = cPickle.load(open(model_fn, 'rb'))
# else:
prelu_alpha = 0.25
mlp = MLP([
    ConvolutionalLayer((5, 5), 32, train_bias=True, pad=0, leaky_relu_alpha=1, max_kernel_norm=.9),
    BatchNormalization(),
    MaxPool((2, 2), (2, 2)),
    PReLU(prelu_alpha),

    # GaussianDropout(0.5),

    ConvolutionalLayer((3, 3), 64, train_bias=True, pad=0, leaky_relu_alpha=prelu_alpha, max_kernel_norm=1.9),
    BatchNormalization(),
    PReLU(prelu_alpha),

    ConvolutionalLayer((3, 3), 64, train_bias=True, pad=0, leaky_relu_alpha=prelu_alpha, max_kernel_norm=1.9),
    BatchNormalization(),
    MaxPool((2, 2), (2, 2)),
    PReLU(prelu_alpha),

    ConvolutionalLayer((3, 3), 128, train_bias=True, pad=0, leaky_relu_alpha=prelu_alpha, max_kernel_norm=1.9),
    BatchNormalization(),
    # MaxPool((2, 2), (2, 2)),
    PReLU(prelu_alpha),

    ConvolutionalLayer((3, 3), 128, train_bias=True, pad=0, leaky_relu_alpha=prelu_alpha, max_kernel_norm=1.9),
    BatchNormalization(),
    MaxPool((2, 2), (2, 2)),
    PReLU(prelu_alpha),

    ConvolutionalLayer((3, 3), 192, train_bias=True, pad=0, leaky_relu_alpha=prelu_alpha, max_kernel_norm=1.9),
    BatchNormalization(),
    MaxPool((2, 2), (2, 2)),
    PReLU(prelu_alpha),

    Flatten(),

    # GaussianDropout(0.5),
    DenseLayer(3000, leaky_relu_alpha=prelu_alpha, max_col_norm=1.9),
    BatchNormalization(),
    # NonLinearity(),
    PReLU(prelu_alpha),

    GaussianDropout(1),
    DenseLayer(3000, leaky_relu_alpha=prelu_alpha, max_col_norm=1.9),
    BatchNormalization(),
    NonLinearity(),
    PReLU(prelu_alpha),

    GaussianDropout(1),
    DenseLayer(len_out, leaky_relu_alpha=prelu_alpha, max_col_norm=1.9),
    BatchNormalization(),
    NonLinearity(activation=T.nnet.softmax)
], (1, 96, 96))



## TODO move to mlp.get_updates
l2 = 0  # 1e-5
learning_rate = np.exp(-1)
momentum = 0.99
epoch_count = 1000
batch_size = 50
minibatch_count = train_x.shape[0] // batch_size
learning_decay = 0.5 ** (1./(800 * minibatch_count))
momentum_decay = 0.5 ** (1./(300 * minibatch_count))
lr_min = 1e-6
mm_min = 0.4
iteration_count = 1

method = 'nesterov'

logger.write('batch=%d, l2=%f, method=%s\nlr=%f, lr_decay=%f,\nmomentum=%f, momentum_decay=%f' %
      (batch_size, l2, method, learning_rate, learning_decay, momentum, momentum_decay))



X = T.tensor4('X4', dtype=floatX)
X.tag.test_value = valid_x[0:1]
Y = T.matrix('Y', dtype=dtype_y)

prob = mlp.forward(X)
cost = mlp.nll(X, Y, l2, train=True)

misclass = theano.function([X, Y], T.eq(T.argmax(prob, axis=1), T.argmax(Y, axis=1)))

nll = theano.function([X, Y], mlp.nll(X, Y, l2, train=False))



lr = theano.shared(np.array(learning_rate, dtype=floatX))
mm = theano.shared(np.array(momentum, dtype=floatX))
updates = mlp.updates(cost, X, momentum=mm, learning_rate=lr, method=method)
updates[lr] = T.maximum(lr * learning_decay, lr_min)
updates[mm] = T.maximum(mm * momentum_decay, mm_min)

train_model = theano.function(
    inputs=[X, Y],
    outputs=cost,
    updates=updates
)

scales = [1.01**(p*abs(p)) for p in map(lambda x: x/2., range(-11, 11))]

randomization_params = {
    'window': (96, 96),
    'scales': scales,
    'angles': range(360),
    'x_offsets': (0, 1),
    'y_offsets': (0, 1),
    # 'median_radii': range(4),
    # 'mean_radii': range(3),
    'flip': True
}
pprint(randomization_params)


def randomize(dataset):
    return randomize_dataset_bc01(dataset, **randomization_params)


r_train_x = randomize(train_x)
indexes = np.arange(train_x.shape[0])
train_start = time.time()
for i in range(epoch_count):
    epoch_start_time = time.time()
    np.random.shuffle(indexes)

    with ProcessPoolExecutor(max_workers=2)as ex:
        valid_futures = ex.map(randomize, repeat(valid_x, iteration_count))
        train_future = ex.submit(randomize, train_x)
        # valid_future = ex.submit(randomize, valid_x)
        batch_nlls = []
        for b in range(minibatch_count):
            k = indexes[b * batch_size:(b + 1) * batch_size]
            batch_x = r_train_x[k]
            batch_y = train_y[k]
            batch_nll = float(train_model(batch_x, batch_y))
            batch_nlls.append(batch_nll)
        train_nll = np.mean(batch_nlls)

        test_nlls = []
        valid_misclasses = []
        # r_valid_x = valid_future.result()
        for r_valid_x in valid_futures:
            # valid_future = ex.submit(randomize, valid_x)
            for vb in range(valid_x.shape[0] // batch_size):
                k = range(vb * batch_size, (vb + 1) * batch_size)
                batch_x = r_valid_x[k]
                batch_y = valid_y[k]
                batch_nll = float(nll(batch_x, batch_y))
                test_nlls.append(batch_nll)
                batch_misclass = misclass(batch_x, batch_y) * 100
                valid_misclasses.append(batch_misclass)
            # r_valid_x = valid_future.result()
        test_nll = np.mean(test_nlls)
        valid_misclass = np.mean(valid_misclasses)
        epoch_time = time.time() - epoch_start_time
        logger.write('%d\t%.5f\t%.5f\t%.1fs\t%.2f%%\t%f\t%f' % (i, train_nll, test_nll, epoch_time, valid_misclass,
                                                                lr.get_value(), mm.get_value()))
        r_train_x = train_future.result()
    cPickle.dump(mlp, open(model_fn, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL)

total_spent_time = time.time() - train_start
logger.write('Trained %d epochs in %.1f seconds (%.2f seconds in average)' % (epoch_count, total_spent_time,
                                                                       total_spent_time / epoch_count))