#!/usr/bin/env python2
#################################
#
# Hand-made ANN library
#
#################################

from __future__ import print_function
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
        raise NotImplementedError('forward is not implemented')

    def get_params(self):
        raise NotImplementedError('get_params not implemented. If no params, then return ()')

    def setup_input(self, input_shape):
        """Returns output_dimension"""
        raise NotImplementedError('setup_input not implemented')

    def self_updates(self):
        return ()


class DenseLayer(ForwardPropogator):
    def __init__(self, features_count, activation=identity):
        self.activation = activation
        self.features_count = features_count

    def setup_input(self, input_shape):
        assert len(input_shape) == 1, 'DenseLayer''s input must be 1 dimensional'
        self.in_dim = np.prod(input_shape)

        self.W = theano.shared(np.cast[theano.config.floatX](np.random.normal(0, np.sqrt(2./self.features_count),
                                                                              (self.in_dim, self.features_count))),
                               borrow=True)
        self.b = theano.shared(np.zeros((1, self.features_count), dtype=theano.config.floatX), borrow=True,
                               broadcastable=(True, False))
        return self.features_count,

    def get_params(self):
        return (self.W, 1), (self.b, 0)

    def forward(self, X):
        return self.activation(T.dot(X, self.W) + self.b)


class PReLU(ForwardPropogator):
    def setup_input(self, input_shape):
        init_vals = np.zeros(input_shape, dtype=theano.config.floatX) + 0.25
        self.alpha = theano.shared(init_vals, borrow=True)
        return input_shape

    def get_params(self):
        return (self.alpha, 0),

    def forward(self, X):
        return T.maximum(0, X) + self.alpha*T.minimum(0, X)


class NonLinearity(ForwardPropogator):
    def __init__(self, activation=ReLU):
        self.activation=activation

    def setup_input(self, input_shape):
        return input_shape

    def get_params(self):
        return ()

    def forward(self, X):
        return self.activation(X)


class NaiveConvBN(ForwardPropogator):
    def __init__(self, eps=1e-12):
        self.eps = eps

    def setup_input(self, input_shape):
        """Assumed input_shape to be ('c', 0, 1)"""
        self.gamma = theano.shared(np.ones((1,) + input_shape, dtype=theano.config.floatX), borrow=True,
                                   broadcastable=(True, False, False, False))
        self.beta = theano.shared(np.zeros((1,) + input_shape, dtype=theano.config.floatX), borrow=True,
                                  broadcastable=(True, False, False, False))
        return input_shape

    def get_params(self):
        return (self.gamma, 1), (self.beta, 1)

    def forward(self, X):
        mean = T.mean(X, axis=0)
        var = T.var(X, axis=0)
        normalized_X = (X - mean) / (var + self.eps)
        return normalized_X * self.gamma + self.beta
        # return X * self.gamma + self.beta


class NaiveBatchNormalization(ForwardPropogator):
    ## TODO Make it not naive
    def __init__(self, eps=1e-12):
        self.eps = eps

    def setup_input(self, input_shape):
        self.gamma = theano.shared(np.ones(input_shape, dtype=theano.config.floatX), borrow=True)
        self.beta = theano.shared(np.zeros(input_shape, dtype=theano.config.floatX), borrow=True)
        return input_shape

    def get_params(self):
        return (self.gamma, 1), (self.beta, 0)

    def forward(self, X):
        mean = T.mean(X, axis=1, keepdims=True)
        std = T.std(X, axis=1, keepdims=True)
        normalized_X = (X - mean) / (std*std + self.eps)
        return normalized_X * self.gamma + self.beta

    # def self_updates(self):
    #     return self.updates


class ConvolutionalLayer(ForwardPropogator):
    def __init__(self, window, features_count, istdev=None, train_bias=True):
        self.features_count = features_count
        self.window = window
        self.istdev = istdev
        self.train_bias = train_bias

    def setup_input(self, input_shape):
        """input_shape=('c', 0, 1)"""
        assert input_shape[-1] == input_shape[-2], 'image must be square'
        img_size = input_shape[-1]
        channels = input_shape[0]
        self.filter_shape = (self.features_count, channels) + self.window

        out_image_size = img_size - self.window[0] + 1
        if self.istdev is None:
            n = np.prod(self.window) * self.features_count
            std = np.sqrt(2./n)
        else:
            std = self.istdev
        self.W = theano.shared(np.cast[theano.config.floatX](np.random.normal(0, std, self.filter_shape)), borrow=True)
        if self.train_bias:
            self.b = theano.shared(np.zeros((1, self.features_count, out_image_size, out_image_size),
                                            dtype=theano.config.floatX),
                                   borrow=True, broadcastable=(True, False, False, False))
            self.params = (self.W, 1), (self.b, 0)
        else:
            self.params = (self.W, 1),
            self.b = 0

        return self.features_count, out_image_size, out_image_size

    def get_params(self):
        return self.params

    def forward(self, X):
        return conv2d(X, self.W, filter_shape=self.filter_shape) + self.b


class MaxPool(ForwardPropogator):
    def __init__(self, window):
        self.window = window

    def setup_input(self, input_shape):
        assert len(input_shape) == 3
        chans, rows, cols = input_shape
        return chans, (rows-1) // self.window[0] + 1, (cols-1) // self.window[1] + 1

    def get_params(self):
        return ()

    def forward(self, X):
        return max_pool_2d(X, ds=self.window)


class Flatten(ForwardPropogator):
    def forward(self, X):
        return T.flatten(X, outdim=2)

    def setup_input(self, input_shape):
        return np.prod(input_shape),

    def get_params(self):
        return ()


####################################
#  Class that contains all layers
####################################


class MLP(ForwardPropogator):
    def __init__(self, layers, input_shape, verbose=True):
        self.layers = layers
        self.input_shape = input_shape
        s = input_shape
        for l in self.layers:
            outp = l.setup_input(s)
            print('%s: %s -> %s' % (l.__class__.__name__, ','.join(map(str, s)), ','.join(map(str, outp))))
            s = outp
        self.output_shape = s

    def forward(self, X):
        for l in self.layers:
            X = l.forward(X)
        return X

    def get_params(self):
        return chain(*[L.get_params() for L in self.layers])

    def get_updates(self, cost, momentum=1.0, learning_rate=0.05):
        updates = []
        for p, l2scale in self.get_params():
            delta_p = theano.shared(p.get_value()*0., broadcastable=p.broadcastable)
            updates.append((p, p - learning_rate*delta_p))
            updates.append((delta_p, momentum*delta_p + (1. - momentum)*T.grad(cost, p)))
        for l in self.layers:
            updates.extend(l.self_updates())
        return updates

    def nll(self, X, Y, l2=0):
        Y1 = self.forward(X)
        Y1 = T.maximum(Y1, 1e-15)
        L = -T.sum(T.log(Y1) * Y) / Y.shape[0]
        for p, l2scale in self.get_params():
            L += T.sum(l2scale * l2 * p * p / 2)
        return L
