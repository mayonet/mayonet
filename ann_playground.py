#!/usr/bin/env python2
from __future__ import division, print_function
from time import time
import math

import numpy as np
from pylearn2.format.target_format import convert_to_one_hot
from spyderlib.utils.qthelpers import MacApplication

import theano
import theano.tensor as T
# theano.config.compute_test_value = 'warn'
theano.config.exception_verbosity = 'high'
theano.config.floatX = 'float32'
theano.config.blas.ldflags = '-lblas -lgfortran'
floatX = theano.config.floatX

from ann import *


np.random.seed(1100)

import os
if not os.path.isfile('mnist.pkl.gz'):
    import urllib
    origin = (
        'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
    )
    print('Downloading data from %s' % origin)
    urllib.urlretrieve(origin, 'mnist.pkl.gz')
import gzip
import cPickle
with gzip.open('mnist.pkl.gz', 'rb') as f:
    train_set, valid_set, test_set = cPickle.load(f)

dtype_y = 'int8'


def to_img(rows, channels_count=1):
    assert len(rows.shape) == 2
    n = rows.shape[0]
    size = math.sqrt(rows.shape[1] // channels_count)
    assert size*size == rows.shape[1] // channels_count
    return np.cast[floatX](rows.reshape(n, channels_count, size, size))

train_x = to_img(train_set[0])
train_y = convert_to_one_hot(train_set[1], dtype=dtype_y)

valid_x = to_img(valid_set[0])
valid_y = convert_to_one_hot(valid_set[1], dtype=dtype_y)

# train_x = np.vstack((train_x, valid_x))
# train_y = np.vstack((train_y, valid_y))

test_x = to_img(test_set[0])
test_y = convert_to_one_hot(test_set[1], dtype=dtype_y)


len_in = train_x.shape[1]
len_out = train_y.shape[1]

prelu_alpha = 0.25

mlp = MLP([
    # ConvolutionalLayer((8, 8), 96, train_bias=True, pad=0, max_kernel_norm=0.9),
    # BatchNormalization(),
    # MaxPool((4, 4), (2, 2)),
    # # NonLinearity(),
    # Maxout(2),
    #
    # # GaussianDropout(0.5),
    #
    # ConvolutionalLayer((8, 8), 96, train_bias=True, pad=3, max_kernel_norm=1.9365),
    # BatchNormalization(),
    # MaxPool((4, 4), (2, 2)),
    # # NonLinearity(),
    # Maxout(2),
    #
    # # GaussianDropout(1),
    #
    # ConvolutionalLayer((5, 5), 48, train_bias=True, pad=3, max_kernel_norm=1.9365),
    # BatchNormalization(),
    # MaxPool((2, 2), (2, 2)),
    # # NonLinearity(),
    # Maxout(2),

    Flatten(),
    Dropout(p=0.8),
    # GaussianDropout(0.5),
    DenseLayer(1200, max_col_norm=1.9365, leaky_relu_alpha=1),
    BatchNormalization(),
    # NonLinearity(),
    # PReLU(prelu_alpha),
    Maxout(pieces=5),

    # GaussianDropout(1),
    DenseLayer(1200, max_col_norm=1.9365, leaky_relu_alpha=2),
    BatchNormalization(),
    # NonLinearity(),
    # PReLU(prelu_alpha),
    # Dropout(p=0.8, w=1),
    Maxout(pieces=5),

    # GaussianDropout(1),
    DenseLayer(10, max_col_norm=1.9365, leaky_relu_alpha=2),
    BatchNormalization(),
    NonLinearity(activation=T.nnet.softmax)
], train_x.shape[1:])


## TODO move to mlp.get_updates
l2 = 0  # 1e-5
learning_rate = 6e-2
momentum = 0.99
epoch_count = 1000
batch_size = 100
minibatch_count = train_x.shape[0] // batch_size
learning_decay = 0.5 ** (1./(800 * minibatch_count))
momentum_decay = 0.5 ** (1./(300 * minibatch_count))
lr_min = 1e-6
mm_min = 0.4

method = 'adadelta'

print('batch=%d, l2=%f, method=%s\nlr=%f, lr_decay=%f,\nmomentum=%f, momentum_decay=%f' %
      (batch_size, l2, method, learning_rate, learning_decay, momentum, momentum_decay))


tr = Trainer(mlp, batch_size, learning_rate, train_x, train_y, valid_X=valid_x, valid_y=valid_y, method=method,
             momentum=momentum, lr_decay=learning_decay, lr_min=lr_min, l2=l2, mm_decay=momentum_decay, mm_min=mm_min,
             model_file_name='mnist.pkl', save_freq=1, epoch_count=epoch_count)

train_start = time.time()
for res in tr():
    print('{epoch}\t{train_nll:.5f}\t{test_nll:.5f}\t{epoch_time:.1f}s\t{valid_misclass:.2f}%%\t{lr}\t'
                 '{momentum}\t{l2_error}'.format(**res))

total_spent_time = time.time() - train_start
print('Trained %d epochs in %.1f seconds (%.2f seconds in average)' % (epoch_count, total_spent_time,
                                                                       total_spent_time / epoch_count))




