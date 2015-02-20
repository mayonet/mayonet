#!/usr/bin/env python2
from __future__ import division, print_function
from time import time

import numpy as np
from pylearn2.format.target_format import convert_to_one_hot

import theano
import theano.tensor as T
theano.config.compute_test_value = 'warn'
theano.config.exception_verbosity = 'high'
theano.config.floatX = 'float32'

from ann import *


np.random.seed(110)

import gzip
import cPickle
with gzip.open('mnist.pkl.gz', 'rb') as f:
    train_set, valid_set, test_set = cPickle.load(f)

dtype_y = 'int32'

train_x = train_set[0]
train_y = convert_to_one_hot(train_set[1], dtype=dtype_y)

valid_x = valid_set[0]
valid_y = convert_to_one_hot(valid_set[1], dtype=dtype_y)

test_x = test_set[0]
test_y = convert_to_one_hot(test_set[1], dtype=dtype_y)


len_in = train_x.shape[1]
len_out = train_y.shape[1]


c1 = ConvolutionalLayer((3, 3), 32)
f = Flatten()
ls = DenseLayer((28-2)**2 * 32, len_out, activation=T.nnet.softmax)
mlp = MLP([c1, f, ls])


# h = 100
# bn0 = NaiveBatchNormalization(len_in, eps=1e-2)
# l0 = DenseLayer(len_in, h)
# a0 = PReLU(h)
# bn1 = NaiveBatchNormalization(h, eps=1e-2)
# l1 = DenseLayer(h, h)
# a1 = PReLU(h)
# bn2 = NaiveBatchNormalization(h, eps=1e-2)
# l2 = DenseLayer(h, h)
# a2 = PReLU(h)
# bn3 = NaiveBatchNormalization(h, eps=1e-2)
# l3 = DenseLayer(h, h)
# a3 = PReLU(h)
# bn4 = NaiveBatchNormalization(h, eps=1e-2)
# l4 = DenseLayer(h, h)
# a4 = PReLU(h)
# bn5 = NaiveBatchNormalization(h, eps=1e-2)
# l5 = DenseLayer(h, h)
# a5 = PReLU(h)
# ls = DenseLayer(h, len_out, activation=T.nnet.softmax)
# mlp = MLP([l0, a0, l1, a1, bn2, l2, a2, bn3, l3, a3, bn4, l4, a4, bn5, l5, a5, ls])
# mlp = MLP([l0, l1, l2, l3, l4, l5, ls])

# good_l0_W = l0.W.get_value(False)
# good_l0_b = l0.b.get_value(False)
# l0.W.set_value(good_l0_W, False)
# l0.b.set_value(good_l0_b, False)


learning_rate = 0.025
momentum = 0.9
epoch_count = 100
batch_size = 100

X = T.matrix('X', dtype=theano.config.floatX)
Y = T.matrix('Y', dtype=dtype_y)

prob = mlp.forward(X)
cost = mlp.nll(X, Y)

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





