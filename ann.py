#!/usr/bin/env python2
#################################
#
# Hand-made ANN library
#
#################################

from itertools import chain
import numpy as np
import theano
import theano.tensor as T


########################################
#  Activation Functions
#########################################
from theano.tensor.nnet import conv2d
from theano.tensor.signal.downsample import max_pool_2d


def identity(X):
    return X

def ReLU(X):
    return T.maximum(0, X)

def LeakyReLU(alpha):
    def ReLU(X):
        return T.maximum(0, X) + alpha * T.minimum(0, X)
    return ReLU


###################################
#   Layers
###################################


class ForwardPropogator:
    def forward(self, X):
        raise NotImplementedError('Do not call functions from abstract class')

    def get_params(self):
        raise NotImplementedError('Do n'
                                  'ot call functions from abstract class')


class DenseLayer(ForwardPropogator):
    def __init__(self, in_dim, out_dim, activation=identity):
        self.activation = activation
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.W = theano.shared(np.cast[theano.config.floatX](np.random.uniform(-0.05, 0.05,
                                                                               (self.in_dim, self.out_dim))),
                               borrow=True)
        self.b = theano.shared(np.zeros((1, self.out_dim), dtype=theano.config.floatX), borrow=True,
                               broadcastable=(True, False))

    def get_params(self):
        return self.W, self.b

    def forward(self, X):
        return self.activation(T.dot(X, self.W) + self.b)


class PReLU(ForwardPropogator):
    def __init__(self, feature_count):
        self.alpha = theano.shared(np.array([0.25] * feature_count, dtype=theano.config.floatX), borrow=True)

    def get_params(self):
        return self.alpha,

    def forward(self, X):
        return T.maximum(0, X) + self.alpha*T.minimum(0, X)


class NaiveBatchNormalization(ForwardPropogator):
    ## TODO Make it not naive
    def __init__(self, input_dimension, eps=1e-12):
        self.gamma = theano.shared(np.ones(input_dimension, dtype=theano.config.floatX), borrow=True)
        self.beta = theano.shared(np.ones(input_dimension, dtype=theano.config.floatX), borrow=True)
        self.eps = eps

    def get_params(self):
        return self.gamma, self.beta

    def forward(self, X):
        mean = T.mean(X, axis=0)
        std = T.std(X, axis=0)
        normalized_X = (X - mean) / (std*std + self.eps)
        return normalized_X * self.gamma + self.beta


class ConvolutionalLayer(ForwardPropogator):
    def __init__(self, window, features_count, batch_size=None):
        axes = (features_count, 1) + window
        # self.image_shape = (batch_size,) + axes if batch_size is not None else None
        self.W = theano.shared(np.cast[theano.config.floatX](np.random.uniform(-0.05, 0.05, size=axes)), borrow=True)
        self.b = theano.shared(np.cast[theano.config.floatX](np.random.uniform(-0.05, 0.05, size=(1, features_count, 1, 1))),
                               borrow=True, broadcastable=(True, False, True, True))

    def get_params(self):
        return self.W, self.b

    def forward(self, X):
        ## TODO image_shape
        return conv2d(X, self.W) + self.b


class MaxPool(ForwardPropogator):
    def __init__(self, window):
        self.window = window

    def get_params(self):
        return ()

    def forward(self, X):
        return max_pool_2d(X, ds=self.window)


class Flatten(ForwardPropogator):
    def forward(self, X):
        return T.flatten(X, 2)

    def get_params(self):
        return ()


####################################
#  Class that contains all layers
####################################


class MLP(ForwardPropogator):
    def __init__(self, layers):
        self.layers = layers

    def forward(self, X):
        for l in self.layers:
            X = l.forward(X)
        return X

    def get_params(self):
        return chain(*[L.get_params() for L in self.layers])

    def get_updates(self, cost, momentum=1.0, learning_rate=0.05):
        updates = []
        for p in self.get_params():
            delta_p = theano.shared(p.get_value()*0., broadcastable=p.broadcastable)
            updates.append((p, p - learning_rate*delta_p))
            updates.append((delta_p, momentum*delta_p + (1. - momentum)*T.grad(cost, p)))
        return updates

    def nll(self, X, Y):
        return -T.sum(T.log(self.forward(X)) * Y) / Y.shape[0]
