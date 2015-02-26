#!/usr/bin/env python2
from __future__ import division, print_function
from time import time
import math

import numpy as np
from pylearn2.format.target_format import convert_to_one_hot

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
    # # GaussianDropout(0.5),
    # ConvolutionalLayer((3, 3), 16, train_bias=True),
    # # ConvBN(),
    # MaxPool((3, 3)),
    # NonLinearity(),
    #
    # # GaussianDropout(0.5),
    # ConvolutionalLayer((3, 3), 32, train_bias=False),
    # ConvBNOnPixels(),
    # MaxPool((3, 3)),
    # NonLinearity(),

    Flatten(),

    # # GaussianDropout(0.5),
    # DenseLayer(128),
    # BatchNormalization(),
    # NonLinearity(),

    # GaussianDropout(1),
    DenseLayer(2048),
    # BatchNormalization(),


    Dropout(0.5),
    NonLinearity(),
    # Maxout(pieces=4),

    # GaussianDropout(1),
    DenseLayer(10),
    # BatchNormalization(),
    NonLinearity(activation=T.nnet.softmax)
], train_x.shape[1:])


## TODO move to mlp.get_updates
l2 = 0  # 1e-5
learning_rate = 1e-2
momentum = 0.99
epoch_count = 500
batch_size = 100
minibatch_count = train_x.shape[0] // batch_size
learning_decay = 0.5 ** (1./(100 * minibatch_count))
momentum_decay = 0.5 ** (1./(300 * minibatch_count))

print('batch=%d, l2=%f,\nlr=%f, lr_decay=%f,\nmomentum=%f, momentum_decay=%f' %
      (learning_rate, batch_size, l2, learning_decay, momentum, momentum_decay))



X = T.tensor4('X4', dtype=floatX)
X.tag.test_value = valid_x[0:1]
Y = T.matrix('Y', dtype=dtype_y)

prob = mlp.forward(X)
cost = mlp.nll(X, Y, l2, train=True)

misclass = theano.function([X, Y], T.sum(prob * Y > 0.5) * 1.0 / Y.shape[0])

nll = theano.function([X, Y], mlp.nll(X, Y, l2, train=False))



lr = theano.shared(np.array(learning_rate, dtype=floatX))
mm = theano.shared(np.array(momentum, dtype=floatX))
updates = mlp.nag_updates(cost, X, momentum=mm, learning_rate=lr)
updates.append([lr, lr * learning_decay])
updates.append([mm, mm * momentum_decay])

train_model = theano.function(
    inputs=[X, Y],
    outputs=cost,
    updates=updates
)


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
    test_nlls = []
    valid_misclasses = []
    for vb in range(valid_x.shape[0] // batch_size):
        k = range(vb * batch_size, (vb + 1) * batch_size)
        batch_x = valid_x[k]
        batch_y = valid_y[k]
        batch_nll = float(nll(batch_x, batch_y))
        test_nlls.append(batch_nll)
        batch_misclass = misclass(batch_x, batch_y) * 100
        valid_misclasses.append(batch_misclass)
    test_nll = np.mean(test_nlls)
    valid_misclass = np.mean(valid_misclasses)
    epoch_time = time() - epoch_start_time
    print('%d\t%.5f\t%.5f\t%.1fs\t%.2f%%\t%f\t%f' % (i, train_nll, test_nll, epoch_time, valid_misclass,
                                                     lr.get_value(), mm.get_value()))
total_spent_time = time() - train_start
print('Trained %d epochs in %.1f seconds (%.2f seconds in average)' % (epoch_count, total_spent_time,
                                                                       total_spent_time / epoch_count))
# mc = '%.1f%%' % (misclass(test_x, test_y)*100)
# print(mc)





