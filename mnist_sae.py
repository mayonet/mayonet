#!/usr/bin/env python2
from __future__ import division, print_function
from time import time
import math

import numpy as np
from pylearn2.format.target_format import convert_to_one_hot
from matplotlib.pyplot import *
import matplotlib.cm as cm
from pylearn2.utils.image import tile_raster_images

import theano
import theano.tensor as T
# theano.config.compute_test_value = 'warn'
from sae import *

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

# train_x = np.vstack((train_x, valid_x))
# train_y = np.vstack((train_y, valid_y))

test_x = to_img(test_set[0])
test_y = convert_to_one_hot(test_set[1], dtype=dtype_y)







input_shape = train_x.shape[1:]
conv = ConvolutionalLayer((3, 3), 16, train_bias=False)
dae = MLP(
    [
        conv,
        NonLinearity(activation=T.nnet.sigmoid),
        DeConv(conv),
        NonLinearity(activation=T.nnet.sigmoid)
    ],
    input_shape)

X = T.tensor4(dtype=floatX)
rec = dae.forward(X, train=True)
reconstruct = theano.function([X], rec)
cost = - T.mean(T.sum(X * T.log(rec) + (1 - X) * T.log(1 - rec), axis=1))
# cost = T.sum(T.sqr(X - rec)) / X.shape[0]

updates = OrderedDict()
learning_rate = theano.shared(np.array(100, dtype=floatX))
for p, l2 in dae.get_params():
    grad = T.grad(cost, p)
    vel = theano.shared(p.get_value()*0., broadcastable=p.broadcastable)
    updates[vel] = 0.5*vel - learning_rate*grad
    # updates[vel] = 0.5*updates[vel] - learning_rate*grad
    updates[p] = p + updates[vel]
train = theano.function([X], cost, updates=updates)

learning_rate.set_value(30)
batch_size = 1000
for epoch in range(30):
    errors = []
    for b in range(train_x.shape[0] // batch_size):
        err = train(train_x[(b * batch_size):((b + 1) * batch_size)])
        # print(err)
        errors.append(err)
    print(epoch, np.mean(errors))


def img(image_c01):
    axis('off')
    imshow(image_c01[0], cmap=cm.Greys_r)


### Draw reconstructed images
ioff()
pics = test_x[0:10]
recs = reconstruct(pics)
for i in range(pics.shape[0]):
    subplot(pics.shape[0], 2, i*2+1)
    img(pics[i])
    subplot(pics.shape[0], 2, i*2+2)
    img(recs[i])
show()
ion()

### Draw features
img(tile_raster_images(conv.W.get_value().T, input_shape[1:], (10, 10), (1, 1))[np.newaxis, :, :])
show()