from ann import *


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