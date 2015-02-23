#!/usr/bin/env python2
from __future__ import division, print_function
from time import time
import math

import numpy as np
from pylearn2.format.target_format import convert_to_one_hot

import theano
import theano.tensor as T
# theano.config.compute_test_value = 'warn'
# theano.config.exception_verbosity = 'high'
theano.config.floatX = 'float32'
theano.config.blas.ldflags = '-lblas -lgfortran'

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

dtype_y = 'int32'


def to_img(rows, channels_count=1):
    assert len(rows.shape) == 2
    n = rows.shape[0]
    size = math.sqrt(rows.shape[1] // channels_count)
    assert size*size == rows.shape[1] // channels_count
    return rows.reshape(n, channels_count, size, size)

train_x = to_img(train_set[0])
train_y = convert_to_one_hot(train_set[1], dtype=dtype_y)

valid_x = to_img(valid_set[0])
valid_y = convert_to_one_hot(valid_set[1], dtype=dtype_y)

test_x = to_img(test_set[0])
test_y = convert_to_one_hot(test_set[1], dtype=dtype_y)


len_in = train_x.shape[1]
len_out = train_y.shape[1]

mlp = MLP([
    ConvolutionalLayer((3, 3), 16, train_bias=False),
    NaiveConvBN(),
    MaxPool((3, 3)),
    NonLinearity(),

    ConvolutionalLayer((3, 3), 32, train_bias=False),
    NaiveConvBN(),
    MaxPool((3, 3)),
    NonLinearity(),

    Flatten(),

    NaiveBatchNormalization(),
    DenseLayer(100),
    NonLinearity(),

    # NaiveBatchNormalization(),
    # DenseLayer(100),
    # NonLinearity(),

    NaiveBatchNormalization(),
    DenseLayer(10),
    NonLinearity(activation=T.nnet.softmax)
], train_x.shape[1:])


# good_l0_W = l0.W.get_value(False)
# good_l0_b = l0.b.get_value(False)
# l0.W.set_value(good_l0_W, False)
# l0.b.set_value(good_l0_b, False)


## TODO move to mlp.get_updates
l2 = 0  # 1e-4
learning_rate = 1e-2
momentum = 0.9
epoch_count = 10
batch_size = 100

print('lr=%f, batch=%d, l2=%f' % (learning_rate, batch_size, l2))



X = T.tensor4('X4', dtype=theano.config.floatX)
X.tag.test_value = valid_x[0:1]
Y = T.matrix('Y', dtype=dtype_y)

prob = mlp.forward(X)
cost = mlp.nll(X, Y, l2)

misclass = theano.function([X, Y], T.sum(prob * Y > 0.5) * 1.0 / Y.shape[0])

nll = theano.function([X, Y], cost)

updates = mlp.get_updates(cost, momentum=momentum, learning_rate=learning_rate)

train_model = theano.function(
    inputs=[X, Y],
    outputs=cost,
    updates=updates
)

minibatch_count = train_x.shape[0] // batch_size
indexes = np.arange(train_x.shape[0])
train_start = time()
for i in range(epoch_count):
    epoch_start_time = time()
    np.random.shuffle(indexes)
    batch_nlls = []
    for b in range(minibatch_count):
        k = indexes[b * batch_size:(b + 1) * batch_size]
        batch_x = train_x[k]
        batch_y = train_y[k]
        batch_nll = float(train_model(batch_x, batch_y))
        batch_nlls.append(batch_nll)
    train_nll = np.mean(batch_nlls)
    test_nll = float(nll(valid_x, valid_y))
    valid_misclass = misclass(valid_x, valid_y) * 100
    epoch_time = time() - epoch_start_time
    print('%d\t%.4f\t%.4f\t%.2fs\t%.1f%%' % (i, train_nll, test_nll, epoch_time, valid_misclass))
total_spent_time = time() - train_start
print('Trained %d epochs in %.1f seconds (%.2f seconds in average)' % (epoch_count, total_spent_time,
                                                                       total_spent_time / epoch_count))
mc = '%.1f%%' % (misclass(test_x, test_y)*100)
print(mc)





