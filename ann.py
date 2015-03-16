#!/usr/bin/env python2
#################################
#
# Hand-made ANN library
#
#################################

from __future__ import print_function
from collections import OrderedDict
from itertools import chain
import itertools
import cPickle
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import sys
import theano

pool_2d = None
try:
    from theano.sandbox.cuda.dnn import dnn_pool
    pool_2d = dnn_pool
except ImportError:
    print("Failed to load 'theano.sandbox.cuda.dnn.dnn_pool'. Loading 'theano.tensor.signal.downsample.max_pool_2d'",
          file=sys.stderr)
    from theano.tensor.signal.downsample import max_pool_2d

    def p_2d(X, ws=(2, 2), stride=None):
        if (stride is not None) and (stride != ws):
            raise RuntimeWarning('theano version of pooling doesn''t support stride other than pool shape')
        return max_pool_2d(X, ds=ws)
    pool_2d = p_2d

import theano.tensor as T
from theano.tensor.nnet import conv2d
import time


########################################
#  Activation Functions
#########################################

def identity(X):
    return X

def ReLU(X):
    return T.maximum(0, X)

def LeakyReLU(alpha):
    def ReLU(X):
        return T.maximum(0, X) + alpha * T.minimum(0, X)
    return ReLU


##################################
#  Helpers
##################################
def calc_norm_numpy(np_W):
    if np_W.ndim == 2:  # DenseLayer case
        return np.sqrt(np.sum(np_W**2, axis=0))
    elif np_W.ndim == 4:  # ConvolutionalLayer case
        return np.sqrt(np.sum(np_W**2, axis=(1, 2, 3)))
    else:
        raise RuntimeError('calc_norm_numpy() works for Dense or Convolutional layers only')


def col_normalize(W, max_col_norm):
    col_norms = T.sqrt(T.sum(T.sqr(W), axis=0))
    desired_norms = T.clip(col_norms, 0, max_col_norm)
    return W * desired_norms / (1e-7 + col_norms)


def kernel_normalize(W, max_kernel_norm):
    kernel_norms = T.sqrt(T.sum(T.sqr(W), axis=(1, 2, 3)))
    desired_norms = T.clip(kernel_norms, 0, max_kernel_norm)
    return W * (desired_norms / (1e-7 + kernel_norms)).dimshuffle(0, 'x', 'x', 'x')

###################################
#   Layers
###################################


class ForwardPropogator:
    def forward(self, X, train=False):
        raise NotImplementedError('forward is not implemented')

    def get_params(self):
        return ()

    def setup_input(self, input_shape):
        """Returns output_dimension"""
        raise NotImplementedError('setup_input not implemented')

    def self_update(self, X, updates):
        pass


class LayerContainer:
    def get_layers(self):
        for prop in self.layers:
            if isinstance(prop, LayerContainer):
                for L in prop.get_layers():
                    yield L
            else:
                yield prop


class Parallel(ForwardPropogator, LayerContainer):
    def __init__(self, propogators):
        self.layers = propogators

    def setup_input(self, input_shapes):
        res = 0
        for fprop, input_shape in zip(self.layers, input_shapes):
            res += np.prod(fprop.setup_input(input_shape))
        return res,

    def forward(self, Xs, train=False):
        return T.concatenate([fprop.forward(X, train) for fprop, X in zip(self.layers, Xs)], axis=1)

    def get_params(self):
        return chain(*(fprop.get_params() for fprop in self.layers))

    def self_update(self, Xs, updates):
        for fprop, X in zip(self.layers, Xs):
            fprop.self_update(X, updates)


class DenseLayer(ForwardPropogator):
    def __init__(self, features_count, max_col_norm=None, activation=identity, leaky_relu_alpha=0,
                 irange=None, lr_scale=1.0):
        self.activation = activation
        self.features_count = features_count
        self.max_col_norm = max_col_norm
        self.leaky_relu_alpha = leaky_relu_alpha
        self.irange = irange
        self.lr_scale = lr_scale

    def setup_input(self, input_shape):
        assert len(input_shape) == 1, 'DenseLayer''s input must be 1 dimensional'
        self.in_dim = np.prod(input_shape)

        if self.irange is None:
            self.W = theano.shared(np.cast[theano.config.floatX](
                np.random.normal(0, np.sqrt(2./((1+self.leaky_relu_alpha**2)*self.features_count)),
                                 (self.in_dim, self.features_count))),
                borrow=True)
        else:
            self.W = theano.shared(np.cast[theano.config.floatX](
                np.random.uniform(-self.irange, self.irange,
                                  (self.in_dim, self.features_count))), borrow=True)
        self.b = theano.shared(np.zeros((1, self.features_count), dtype=theano.config.floatX), borrow=True,
                               broadcastable=(True, False))
        return self.features_count,

    def get_params(self):
        return {'param': self.W, 'l2_scale': 1.0, 'lr_scale': getattr(self, 'lr_scale', 1.)}, \
               {'param': self.b, 'l2_scale': 0.0, 'lr_scale': getattr(self, 'lr_scale', 1.)}

    def forward(self, X, train=False):
        return self.activation(T.dot(X, self.W) + self.b)

    def self_update(self, X, updates):
        if self.max_col_norm is not None:
            W = updates[self.W]
            updates[self.W] = col_normalize(W, self.max_col_norm)


class DenseDecoder(ForwardPropogator):
    def __init__(self, encoding_layer, untied_bias=True):
        self.encoding_layer = encoding_layer
        self.untied_bias = untied_bias

    def get_params(self):
        return self.params

    def setup_input(self, input_shape):
        self.W_prime = self.encoding_layer.W.T
        if self.untied_bias:
            self.b_prime = theano.shared(np.zeros((1, self.encoding_layer.W.get_value().shape[0]), dtype=theano.config.floatX), borrow=True,
                                         broadcastable=(True, False))
            self.params = {'param': self.b_prime, 'l2_scale': 0},
        else:
            self.b_prime = 0
            self.params = ()
        return self.encoding_layer.W.get_value().shape[:1]

    def forward(self, X, train=False):
        return T.dot(X, self.W_prime) + self.b_prime


class DeConv(ForwardPropogator):
    def __init__(self, mirror_layer, train_bias=True, pad=0,
                 max_kernel_norm=None):
        self.mirror = mirror_layer
        self.window = self.mirror.window
        self.train_bias = train_bias
        self.pad = pad
        self.max_kernel_norm = max_kernel_norm

    def get_params(self):
        return self.params

    def setup_input(self, input_shape):
        """input_shape=('c', 0, 1)"""
        assert input_shape[-1] == input_shape[-2], 'image must be square'
        img_size = input_shape[-1]
        channels = input_shape[0]
        self.features_count = self.mirror.channels
        self.filter_shape = (self.features_count, channels) + self.window

        out_image_size = img_size+self.pad*2 + self.window[0] - 1
        if self.train_bias:
            self.b = theano.shared(np.zeros((1, self.features_count, 1, 1), dtype=theano.config.floatX),
                                   borrow=True, broadcastable=(True, False, True, True))
            self.params = (self.b, 0),
        else:
            self.params = ()
            self.b = 0

        self.output_shape = self.features_count, out_image_size, out_image_size
        return self.output_shape

    def forward(self, X, train=False):
        return conv2d(X, self.mirror.W.dimshuffle(1, 0, 2, 3), border_mode='full') + self.b


class PReLU(ForwardPropogator):
    def __init__(self, initial_alpha=0.25):
        self.initial_alpha = initial_alpha

    def setup_input(self, input_shape):
        init_vals = np.zeros(input_shape, dtype=theano.config.floatX) + self.initial_alpha
        self.alpha = theano.shared(init_vals, borrow=True)
        return input_shape

    def get_params(self):
        return {'param': self.alpha, 'l2_scale': 0},

    def forward(self, X, train=False):
        return T.maximum(0, X) + self.alpha*T.minimum(0, X)


class NonLinearity(ForwardPropogator):
    def __init__(self, activation=ReLU):
        self.activation=activation

    def setup_input(self, input_shape):
        return input_shape

    def forward(self, X, train=False):
        return self.activation(X)


class LCNNormalization(ForwardPropogator):
    def __init__(self, kernel_size, threshold=1e-7, use_divisor=False):
        def gaussian_filter(kernel_shape):

            x = np.zeros((kernel_shape, kernel_shape), dtype='float32')

            def gauss(x, y, sigma=2.0):
                Z = 2 * np.pi * sigma**2
                return 1./Z * np.exp(-(x**2 + y**2) / (2. * sigma**2))

            mid = np.floor(kernel_shape/2.)
            for i in xrange(0, kernel_shape):
                for j in xrange(0, kernel_shape):
                    x[i, j] = gauss(i-mid, j-mid)

            return (x / sum(x)).reshape((1, 1, kernel_shape, kernel_shape))

        self.kernel_size = kernel_size
        self.filter_shape = (1, 1, kernel_size, kernel_size)
        self.threshold = threshold
        self.use_divisor = use_divisor
        self.filter = theano.shared(np.asarray(gaussian_filter(self.kernel_size), dtype=theano.config.floatX),
                                    borrow=True, broadcastable=(True, True, False, False))

    def setup_input(self, input_shape):
        self.shape = input_shape
        return self.shape

    def forward(self, X, train=False):
        batch_size = X.shape[0]
        X = T.reshape(X, (batch_size*self.shape[0], 1, self.shape[1], self.shape[2]))
        convout = conv2d(X, filters=self.filter, filter_shape=self.filter_shape, border_mode='full')
        mid = int(np.floor(self.kernel_size/2.))
        new_X = X - convout[:, :, mid:-mid, mid:-mid]

        if self.use_divisor:
            # Scale down norm of kernel_sizexkernel_size patch
            sum_sqr_XX = conv2d(T.sqr(T.abs_(X)), filters=self.filter,
                                filter_shape=self.filter_shape, border_mode='full')

            denom = T.sqrt(sum_sqr_XX[:, :, mid:-mid, mid:-mid])
            per_img_mean = denom.mean(axis=[2, 3])
            divisor = T.largest(per_img_mean.dimshuffle(0, 1, 'x', 'x'), denom)
            divisor = T.maximum(divisor, self.threshold)

            new_X /= divisor

        return T.reshape(new_X, ((batch_size,) + self.shape))


class BatchNormalization(ForwardPropogator):
    def __init__(self, eps=1e-12, alpha=0.9, alpha_updates=0.1):
        self.eps = eps
        self.alpha = alpha
        self.alpha_updates = alpha_updates

    def setup_input(self, input_shape):
        bc = (True, ) + (False, ) * len(input_shape)
        self.beta = theano.shared(np.zeros((1, ) + input_shape, dtype=theano.config.floatX),
                                  borrow=True, broadcastable=bc)
        self.gamma = theano.shared(np.ones((1, ) + input_shape, dtype=theano.config.floatX),
                                   borrow=True, broadcastable=bc)
        self.MA = theano.shared(np.zeros((1, ) + input_shape, dtype=theano.config.floatX),
                                borrow=True, broadcastable=bc)
        self.MV = theano.shared(np.ones((1, ) + input_shape, dtype=theano.config.floatX),
                                borrow=True, broadcastable=bc)
        return input_shape

    def get_params(self):
        return {'param': self.gamma, 'l2_scale': 1}, {'param': self.beta, 'l2_scale': 0}

    def forward(self, X, train=False):
        mean = self.MA
        var = self.MV
        if train:
            mean = T.mean(X, axis=0, keepdims=True)*self.alpha + mean*(1-self.alpha)
            var = T.var(X, axis=0, keepdims=True)*self.alpha + var*(1-self.alpha)

        normalized_X = (X - mean) / T.sqrt(var + self.eps)
        return normalized_X * self.gamma + self.beta

    def self_update(self, X, updates):
        updates[self.MA] = T.mean(X, axis=0, keepdims=True)*self.alpha_updates + self.MA*(1-self.alpha_updates)
        updates[self.MV] = T.var(X, axis=0, keepdims=True)*self.alpha_updates + self.MV*(1-self.alpha_updates)


class ConvBNOnChannel(ForwardPropogator):
    def __init__(self, eps=1e-12, alpha=0.9, alpha_updates=0.1):
        self.eps = eps
        self.alpha = alpha
        self.alpha_updates = alpha_updates

    def setup_input(self, input_shape):
        """Assumed input_shape to be ('c', 0, 1)"""
        self.gamma = theano.shared(np.ones((1,) + input_shape, dtype=theano.config.floatX),
                                   borrow=True, broadcastable=(True, False, False, False))
        self.beta = theano.shared(np.zeros((1,) + input_shape, dtype=theano.config.floatX),
                                  borrow=True, broadcastable=(True, False, False, False))
        self.MA = theano.shared(np.zeros((1, 1) + input_shape[1:], dtype=theano.config.floatX),
                                borrow=True, broadcastable=(True, True, False, False))
        self.MV = theano.shared(np.ones((1, 1) + input_shape[1:], dtype=theano.config.floatX),
                                borrow=True, broadcastable=(True, True, False, False))
        return input_shape

    def get_params(self):
        return {'param': self.gamma, 'l2_scale': 1}, {'param': self.beta, 'l2_scale': 0}

    def forward(self, X, train=False):
        mean = self.MA
        var = self.MV
        if train:
            mean = T.mean(X, axis=(0, 1), keepdims=True)*self.alpha + mean*(1-self.alpha)
            var = T.var(X, axis=(0, 1), keepdims=True)*self.alpha + var*(1-self.alpha)
        normalized_X = (X - mean) / T.sqrt(var + self.eps)
        return normalized_X * self.gamma + self.beta

    def self_update(self, X, updates):
        updates[self.MA] = T.mean(X, axis=(0, 1), keepdims=True)*self.alpha_updates + self.MA*(1-self.alpha_updates)
        updates[self.MV] = T.var(X, axis=(0, 1), keepdims=True)*self.alpha_updates + self.MV*(1-self.alpha_updates)


class ConvBNOnPixels(ForwardPropogator):
    def __init__(self, eps=1e-12, alpha=0.9, alpha_updates=0.1):
        self.eps = eps
        self.alpha = alpha
        self.alpha_updates = alpha_updates

    def setup_input(self, input_shape):
        """Assumed input_shape to be ('c', 0, 1)"""
        self.gamma = theano.shared(np.ones((1,) + input_shape, dtype=theano.config.floatX),
                                   borrow=True, broadcastable=(True, False, False, False))
        self.beta = theano.shared(np.zeros((1,) + input_shape, dtype=theano.config.floatX),
                                  borrow=True, broadcastable=(True, False, False, False))
        self.MA = theano.shared(np.zeros((1, input_shape[0], 1, 1), dtype=theano.config.floatX),
                                borrow=True, broadcastable=(True, False, True, True))
        self.MV = theano.shared(np.ones((1, input_shape[0], 1, 1), dtype=theano.config.floatX),
                                borrow=True, broadcastable=(True, False, True, True))
        return input_shape

    def get_params(self):
        return {'param': self.gamma, 'l2_scale': 1}, {'param': self.beta, 'l2_scale': 0}

    def forward(self, X, train=False):
        mean = self.MA
        var = self.MV
        if train:
            mean = T.mean(X, axis=(0, 2, 3), keepdims=True)*self.alpha + mean*(1-self.alpha)
            var = T.var(X, axis=(0, 2, 3), keepdims=True)*self.alpha + var*(1-self.alpha)
        normalized_X = (X - mean) / T.sqrt(var + self.eps)
        return normalized_X * self.gamma + self.beta

    def self_update(self, X, updates):
        updates[self.MA] = T.mean(X, axis=(0, 2, 3), keepdims=True)*self.alpha_updates + self.MA*(1-self.alpha_updates)
        updates[self.MV] = T.var(X, axis=(0, 2, 3), keepdims=True)*self.alpha_updates + self.MV*(1-self.alpha_updates)


class ConvolutionalLayer(ForwardPropogator):
    def __init__(self, window, features_count, train_bias=True, border_mode='valid', pad=0,
                 max_kernel_norm=None, leaky_relu_alpha=0, irange=None, stride=(1, 1)):
        assert border_mode in ('valid', 'full'), 'Border mode must be "valid" or "full"'
        self.features_count = features_count
        self.window = window
        self.train_bias = train_bias
        self.border_mode = border_mode
        self.pad = pad
        self.max_kernel_norm = max_kernel_norm
        self.leaky_relu_alpha = leaky_relu_alpha
        self.irange = irange
        self.stride = stride

    def setup_input(self, input_shape):
        """input_shape=('c', 0, 1)"""
        assert input_shape[-1] == input_shape[-2], 'image must be square'
        img_size = input_shape[-1]
        self.channels = input_shape[0]
        self.filter_shape = (self.features_count, self.channels) + self.window

        stride = self.stride[0]
        out_image_size = (img_size+self.pad*2) // stride
        if self.border_mode == 'valid':
            out_image_size += -self.window[0] + 1
        if self.border_mode == 'full':
            out_image_size += self.window[0] - 1
        n = np.prod(self.window) * self.features_count
        if self.irange is None:
            std = np.sqrt(2./((1+self.leaky_relu_alpha**2)*n))
            self.W = theano.shared(np.cast[theano.config.floatX](np.random.normal(0, std, self.filter_shape)),
                                   borrow=True)
        else:
            self.W = theano.shared(np.cast[theano.config.floatX](np.random.uniform(-self.irange, self.irange,
                                                                 self.filter_shape)), borrow=True)
        if self.train_bias:
            self.b = theano.shared(np.zeros((1, self.features_count, out_image_size, out_image_size),
                                            dtype=theano.config.floatX),
                                   borrow=True, broadcastable=(True, False, False, False))
            self.params = {'param': self.W, 'l2_scale': 1}, {'param': self.b, 'l2_scale': 0},
        else:
            self.params = {'param': self.W, 'l2_scale': 1},
            self.b = 0

        self.output_shape = self.features_count, out_image_size, out_image_size
        return self.output_shape

    def get_params(self):
        return self.params

    def forward(self, X, train=False):
        X0 = T.zeros((X.shape[0], X.shape[1], X.shape[2]+self.pad*2, X.shape[3]+self.pad*2))
        X0 = T.set_subtensor(X0[:, :, self.pad:X.shape[2]+self.pad, self.pad:X.shape[3]+self.pad], X)
        return conv2d(X0, self.W, filter_shape=self.filter_shape, border_mode=self.border_mode, subsample=self.stride) + self.b

    def self_update(self, X, updates):
        if self.max_kernel_norm is not None:
            W = updates[self.W]
            updates[self.W] = kernel_normalize(W, self.max_kernel_norm)


class MaxPool(ForwardPropogator):
    def __init__(self, window, stride=None):
        self.window = window
        self.stride = stride
        if self.stride is None:
            self.stride = self.window

    def setup_input(self, input_shape):
        assert len(input_shape) == 3
        chans, rows, cols = input_shape
        # return chans, (rows-1) // self.window[0] + 1, (cols-1) // self.window[1] + 1
        return chans, (rows-self.window[0]) // self.stride[0] + 1, (cols-self.window[0]) // self.stride[0] + 1

    def forward(self, X, train=False):
        return pool_2d(X, ws=self.window, stride=self.stride)


class Flatten(ForwardPropogator):
    def forward(self, X, train=False):
        return T.flatten(X, outdim=2)

    def setup_input(self, input_shape):
        return np.prod(input_shape),


class Reshape(ForwardPropogator):
    def __init__(self, new_shape):
        self.shape = new_shape

    def setup_input(self, input_shape):
        return self.shape

    def forward(self, X, train=False):
        batch_size = X.shape[0]
        return T.reshape(X, (batch_size,) + self.shape)


class Dropout(ForwardPropogator):
    def __init__(self, p, w=None):
        self.p = p
        self.w = w
        if w is None:
            self.w = 1./p

    def forward(self, X, train=False):
        if train:
            return X*self.mask.binomial(size=self.shape, n=1, p=self.p, dtype=theano.config.floatX)*self.w
        else:
            return X

    def setup_input(self, input_shape):
        self.shape = input_shape
        self.mask = T.shared_randomstreams.RandomStreams()
        return input_shape


class GaussianDropout(ForwardPropogator):
    def __init__(self, std):
        self.std = std

    def forward(self, X, train=False):
        if train:
            return X*self.mask.normal(size=self.shape, avg=1, std=self.std, dtype=theano.config.floatX)
        else:
            return X

    def setup_input(self, input_shape):
        self.shape = input_shape
        self.mask = T.shared_randomstreams.RandomStreams()
        return input_shape


class Maxout(ForwardPropogator):
    def __init__(self, pieces=2,
                 # , pool_stride=None
                 min_zero=False
                 ):
        self.pieces = pieces
        self.min_zero = min_zero
        # self.pool_stride = self.pieces if pool_stride is None else pool_stride

    def setup_input(self, input_shape):
        assert (input_shape[0] % self.pieces) == 0, 'input_shape must be divisible by pieces count'
        self.input_shape = input_shape
        self.output_num = self.input_shape[0] // self.pieces
        return (self.output_num,) + self.input_shape[1:]

    def forward(self, X, train=False):
        Xs = T.reshape(X, (X.shape[0], self.pieces, self.output_num) + self.input_shape[1:])
        Xs = T.max(Xs, axis=1)
        if self.min_zero:
            Xs = T.maximum(Xs, 0.)
        return Xs


####################################
#  Class that contains all layers
####################################


class MLP(ForwardPropogator, LayerContainer):
    def __init__(self, layers, input_shape=None, logger=sys.stdout):
        self.layers = layers
        self.logger = logger
        if input_shape is not None:
            self.input_shape = input_shape
            self.setup_input(self.input_shape)

    def setup_input(self, input_shape):
        s = input_shape
        for l in self.layers:
            outp = l.setup_input(s)
            print('%s: %s -> %s' % (l.__class__.__name__, ','.join(map(str, s)), ','.join(map(str, outp))),
                  file=self.logger)
            s = outp
        self.output_shape = s
        return self.output_shape

    def forward(self, X, train=False):
        if isinstance(X, (list, tuple)) and len(X) == 1:
            X = X[0]
        for l in self.layers:
            X = l.forward(X, train)
        return X

    def get_params(self):
        return chain(*[L.get_params() for L in self.layers])

    def updates(self, cost, l2_error, X, momentum=1.0, learning_rate=0.05, method='momentum', l2=0):
        updates = OrderedDict()
        for p_info in self.get_params():
            if isinstance(p_info, dict):
                p = p_info['param']
                lr_scale = p_info['lr_scale'] if 'lr_scale' in p_info else 1.0
            else:
                p, _ = p_info
                lr_scale = 1.0
            lr = lr_scale * learning_rate

            grad = T.grad(cost + l2*l2_error, p)
            if method == 'sgd':
                updates[p] = p - lr*grad
            elif method == 'adagrad':
                grad_acc = theano.shared(np.zeros(p.get_value().shape, dtype=theano.config.floatX),
                                         broadcastable=p.broadcastable, borrow=True)
                updates[grad_acc] = grad_acc + grad**2
                lr_p = T.clip(lr/T.sqrt(updates[grad_acc] + 1e-7), 1e-6, 50)
                updates[p] = p - lr_p*grad
            elif method == 'adadelta':
                grad_acc = theano.shared(np.zeros(p.get_value().shape, dtype=theano.config.floatX),
                                         broadcastable=p.broadcastable, borrow=True)
                weight_acc = theano.shared(np.zeros(p.get_value().shape, dtype=theano.config.floatX),
                                           broadcastable=p.broadcastable, borrow=True)
                updates[grad_acc] = 0.9*grad_acc + 0.1*(grad**2)
                lr_p = lr*T.sqrt(weight_acc + 1e-7)/T.sqrt(updates[grad_acc] + 1e-7)
                delta_p = -lr_p*grad
                updates[p] = p + delta_p
                updates[weight_acc] = 0.9*weight_acc + 0.1*delta_p**2
            elif method == 'adadelta+nesterov':
                grad_acc = theano.shared(np.zeros(p.get_value().shape, dtype=theano.config.floatX),
                                         broadcastable=p.broadcastable, borrow=True)
                weight_acc = theano.shared(np.zeros(p.get_value().shape, dtype=theano.config.floatX),
                                           broadcastable=p.broadcastable, borrow=True)
                vel = theano.shared(p.get_value()*0., broadcastable=p.broadcastable)
                updates[grad_acc] = 0.9*grad_acc + 0.1*(grad**2)
                lr_p = lr*T.sqrt(weight_acc + 1e-7)/T.sqrt(updates[grad_acc] + 1e-7)
                updates[vel] = momentum*vel - lr_p*grad
                updates[vel] = momentum*updates[vel] - lr_p*grad
                updates[p] = p + updates[vel]
                updates[weight_acc] = 0.9*weight_acc + 0.1*updates[vel]**2
            elif method == 'rmsprop':
                mean_square = theano.shared(np.zeros(p.get_value().shape, dtype=theano.config.floatX),
                                            broadcastable=p.broadcastable, borrow=True)
                updates[mean_square] = 0.9*mean_square + 0.1*(grad**2)
                lr_p = T.clip(lr/T.sqrt(updates[mean_square] + 1e-7), 1e-6, 50)
                updates[p] = p - lr_p*grad
            # elif method == 'esgd':
            #     D = theano.shared(np.zeros(p.get_value().shape, dtype=theano.config.floatX),
            #                                 broadcastable=p.broadcastable)
            #     r = T.shared_randomstreams.RandomStreams()
            #     v = r.normal(size=(grad.shape[1],), avg=0, std=1, dtype=theano.config.floatX)
            #     i = theano.shared(np.array(0, dtype='uint8'))
            #     updates[i] = i+1
            #     updates[D] = D + (T.Rop(grad, p, v))**2
            #     updates[p] = p - learning_rate*grad/(T.sqrt(D/updates[i]) + 1e-2)

            elif method in ('momentum', 'nesterov'):
                vel = theano.shared(p.get_value()*0., broadcastable=p.broadcastable, borrow=True)
                updates[vel] = momentum*vel - lr*grad
                if method == 'nesterov':
                    updates[vel] = momentum*updates[vel] - lr*grad
                updates[p] = p + updates[vel]
            else:
                raise AssertionError('invalid method: %s' % method)
        self.self_update(X, updates)
        return updates

    def self_update(self, X, updates):
        if isinstance(X, (list, tuple)) and len(X) == 1:
            X = X[0]
        for l in self.layers:
            l.self_update(X, updates)
            X = l.forward(X, train=True)


    # def nag_updates(self, cost, X, momentum=1.0, learning_rate=0.05):
    #     return self.sgd_updates(cost, X, momentum, learning_rate, method='nesterov')


def neg_log_likelihood(mlp, X, Y, train=False):
    Y1 = mlp.forward(X, train)
    Y1 = T.maximum(Y1, 1e-15)
    return -T.sum(T.log(Y1) * Y) / Y.shape[0]


def soft_log_likelihood(mlp, X, Y, train=False):
    Y1 = mlp.forward(X, train)
    Y1 = T.maximum(Y1, 1e-15)
    return T.sum(T.log(Y/Y1) * Y) / Y.shape[0]


def models_l2_error(mlp):
    L = 0
    for p_info in mlp.get_params():
        if isinstance(p_info, dict):
            p = p_info['param']
            l2scale = p_info['l2_scale'] if 'l2_scale' in p_info else 1.0
        else:
            p, l2scale = p_info
        if l2scale < 1e-12:
            continue
        L += T.sum(l2scale * p * p / 2)
    return L


def Trainer(model, batch_size, learning_rate, train_X, train_y, valid_X=None, valid_y=None, method='sgd',
            momentum=0, lr_decay=1, lr_min=1e-9, l2=0, mm_decay=1, mm_min=1e-9,
            train_augmentation=identity, valid_augmentation=identity, valid_aug_count=1,
            train_y_augmentation=identity, valid_y_augmentation=identity,
            model_file_name=None, save_freq=None, save_in_different_files=False, epoch_count=None,
            cost_f=neg_log_likelihood, valid_cost_f=None):

    floatX = theano.config.floatX
    minibatch_count = train_y.shape[0] // batch_size
    if not isinstance(train_X, tuple):
        train_X = train_X,
        valid_X = valid_X,

    X = []
    for tr in train_X:
        if tr.ndim == 2:
            X.append(T.matrix('X', dtype=floatX))
        elif tr.ndim == 4:
            X.append(T.tensor4('X', dtype=floatX))
        else:
            raise RuntimeError('Only input with ndim in (2,4) is allowed')
    Y = T.matrix('Y', dtype=floatX)
    prob = model.forward(X)
    cost = cost_f(model, X, Y, train=True)
    l2_error = models_l2_error(model)

    if valid_X is None:
        valid_X = train_X
    if valid_y is None:
        valid_y = train_y

    misclass = theano.function(X + [Y], T.eq(T.argmax(prob, axis=1), T.argmax(Y, axis=1)))
    if valid_cost_f is None:
        valid_cost_f = cost_f
    nll = theano.function(X + [Y], valid_cost_f(model, X, Y, train=False))
    calc_l2_error = theano.function([], l2_error)

    lr = theano.shared(np.array(learning_rate, dtype=floatX))
    mm = theano.shared(np.array(momentum, dtype=floatX))
    updates = model.updates(cost, l2_error, X, momentum=mm, learning_rate=lr, method=method, l2=l2)
    updates[lr] = T.maximum(lr * lr_decay, lr_min)
    updates[mm] = T.maximum(mm * mm_decay, mm_min)

    train_model = theano.function(
        inputs=X + [Y],
        outputs=cost,
        updates=updates
    )

    def trainer_func():
        # r_train_x = train_augmentation(train_X)
        indexes = np.arange(train_y.shape[0])
        best_v_nll = 2.
        if epoch_count is not None:
            iterator = range(epoch_count)
        else:
            iterator = itertools.count()
        for i in iterator:
            epoch_start_time = time.time()
            np.random.shuffle(indexes)

            epoch_train_y = train_y_augmentation(train_y)
            epoch_valid_y = valid_y_augmentation(valid_y)

            with ProcessPoolExecutor(max_workers=2)as ex:
                valid_futures = ex.map(valid_augmentation, itertools.repeat(valid_X, valid_aug_count))
                # train_future = ex.submit(train_augmentation, train_X)
                batch_nlls = []
                k_next = indexes[:batch_size]
                batch_future = ex.submit(train_augmentation, [d[k_next] for d in train_X])
                for b in range(1, minibatch_count+1):
                    k = k_next
                    k_next = indexes[b * batch_size:(b + 1) * batch_size]
                    batch_x = batch_future.result()
                    if len(k_next) > 0:
                        batch_future = ex.submit(train_augmentation, [d[k_next] for d in train_X])
                    batch_y = epoch_train_y[k]
                    batch_nll = float(train_model(*(batch_x + [batch_y])))
                    batch_nlls.append(batch_nll)
                train_nll = np.mean(batch_nlls)
                # del r_train_x  # Try to free up some memory

                test_nlls = []
                valid_misclasses = []
                for r_valid_x in valid_futures:
                    for vb in range(epoch_valid_y.shape[0] // batch_size):
                        k = range(vb * batch_size, (vb + 1) * batch_size)
                        batch_x = [d[k] for d in r_valid_x]
                        batch_y = epoch_valid_y[k]
                        batch_nll = float(nll(*batch_x + [batch_y]))
                        test_nlls.append(batch_nll)
                        batch_misclass = misclass(*batch_x + [batch_y]) * 100
                        valid_misclasses.append(batch_misclass)
                test_nll = np.mean(test_nlls)
                if (test_nll < best_v_nll) and (model_file_name is not None):
                    best_v_nll = test_nll
                    cPickle.dump(model, open('best_' + model_file_name, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL)
                valid_misclass = np.mean(valid_misclasses)
                epoch_time = time.time() - epoch_start_time

                layer_info = OrderedDict()
                for L_id, L in enumerate(model.get_layers()):
                    l_name = '%d%s' % (L_id, L.__class__.__name__)
                    if isinstance(L, DenseLayer) or isinstance(L, ConvolutionalLayer):
                        layer_info[l_name + '_max_norm'] = np.max(calc_norm_numpy(L.W.get_value()))
                        layer_info[l_name + '_mean_norm'] = np.mean(calc_norm_numpy(L.W.get_value()))
                        layer_info[l_name + '_min_norm'] = np.min(calc_norm_numpy(L.W.get_value()))
                        layer_info[l_name + '_l2'] = np.sum(L.W.get_value()**2)
                res = OrderedDict((('epoch', i),
                                   ('epoch_time', epoch_time),
                                   ('train_nll', train_nll),
                                   ('test_nll', test_nll),
                                   ('valid_misclass', valid_misclass),
                                   ('lr', float(lr.get_value())),
                                   ('momentum', float(mm.get_value())),
                                   ('l2_error', calc_l2_error()*l2)))
                res.update(layer_info)
                yield res
                # r_train_x = train_future.result()
            if save_freq is not None and save_freq > 0 and i % save_freq == 0:
                if save_in_different_files:
                    cPickle.dump(model, open('junk/%i_%s' % (i, model_file_name), 'wb'), protocol=cPickle.HIGHEST_PROTOCOL)
                else:
                    cPickle.dump(model, open(model_file_name, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL)

    return trainer_func