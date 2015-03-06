from theano.sandbox.cuda.basic_ops import gpu_contiguous
from ann import *

class Conv2DNoBiasLayer(object):
    def __init__(self,
                 input_layer,
                 n_filters,
                 filter_size,
                 weights_std,
                 stride=1,
                 nonlinearity=ReLU,
                 dropout=0.,
                 partial_sum=None,
                 pad=0,
                 trainable=True):
        """
        Only the valid border mode is supported.
        n_filters should be a multiple of 16
        """
        self.input_layer = input_layer
        self.input_shape = self.input_layer.get_output_shape()
        self.n_filters = n_filters
        n_channels = self.input_shape[0]
        self.n_channels = n_channels
        self.filter_size = filter_size
        self.weights_std = np.float32(weights_std)
        self.stride = stride
        self.nonlinearity = nonlinearity
        self.dropout = dropout
        self.partial_sum = partial_sum
        self.pad = pad
        self.mb_size = self.input_layer.mb_size

        self.data_order = layers.data_order.type2

        assert (len(self.input_layer.get_output_shape()) == 4), \
            'Input must have 4 dimensions.'

        assert (self.input_layer.data_order == self.data_order), \
            'Input data order does not match this layer\'s data order.'

        self.filter_shape = (n_channels, filter_size, filter_size, n_filters)

        self.trainable = trainable
        self.W = layers.shared_single(4)

        self.params = [self.W]
        self.reset_params()

        self.filter_acts_op = FilterActs(stride=self.stride,
                                         partial_sum=self.partial_sum,
                                         pad=self.pad)

    def reset_params(self):
        self.W.set_value(np.random.randn(*self.filter_shape).astype(
            np.float32) * self.weights_std)

    def get_output_shape(self):
        output_width = int(np.ceil((
            self.input_shape[1] + 2 * self.pad - self.filter_size
            + self.stride)*1.0 / self.stride))
        output_height = int(np.ceil((
            self.input_shape[2] + 2 * self.pad - self.filter_size
            + self.stride)*1.0 / self.stride))
        output_shape = (self.n_filters, output_width, output_height,
                        self.mb_size)
        return output_shape

    def output(self, input=None, dropout_active=True, *args, **kwargs):
        if input is None:
            input = self.input_layer.output(dropout_active=dropout_active,
                                            *args, **kwargs)

        if dropout_active and (self.dropout > 0.):
            retain_prob = 1 - self.dropout
            mask = layers.srng.binomial(input.shape, p=retain_prob,
                                        dtype='int32').astype('float32')
            # apply the input mask and rescale the input accordingly.
            # By doing this it's no longer necessary to rescale the weights
            # at test time.
            input = input / retain_prob * mask

        contiguous_input = gpu_contiguous(input)
        contiguous_filters = gpu_contiguous(self.W)
        conved = self.filter_acts_op(contiguous_input, contiguous_filters)

        return self.nonlinearity(conved)


class Deconv2DNoBiasLayer(object):
    def __init__(self,
                 input_layer,
                 mirror_layer,
                 nonlinearity=None):
        """
        Only the valid border mode is supported.
        n_filters should be a multiple of 16
        """

        self.mirror_layer = mirror_layer

        self.input_layer = input_layer
        self.input_shape = self.input_layer.get_output_shape()
        n_filters = self.input_shape[0]

        if nonlinearity:
            self.nonlinearity = nonlinearity
        else:
            self.nonlinearity = mirror_layer.nonlinearity

        self.n_channels = mirror_layer.n_channels
        self.n_filters = mirror_layer.n_filters
        self.filter_size = mirror_layer.filter_size
        self.weights_std = mirror_layer.weights_std
        self.stride = mirror_layer.stride
        self.dropout = mirror_layer.dropout
        self.partial_sum = mirror_layer.partial_sum
        self.pad = mirror_layer.pad
        self.mb_size = self.input_layer.mb_size

        self.filter_shape = mirror_layer.filter_shape

        self.trainable = False
        self.W = mirror_layer.W

        self.params = []

        self.data_order = layers.data_order.type2

        assert (len(self.input_layer.get_output_shape()) == 4), \
            'Input must have 4 dimensions.'

        assert (self.input_layer.data_order == self.data_order), \
            'Input data order does not match this layer\'s data order.'

        self.image_acts_op = ImageActs(stride=self.stride,
                                       partial_sum=self.partial_sum,
                                       pad=self.pad)

    def get_output_shape(self):
        output_shape = self.mirror_layer.input_layer.get_output_shape()
        return output_shape

    def output(self, input=None, dropout_active=True, *args, **kwargs):
        if input is None:
            input = self.input_layer.output(dropout_active=dropout_active,
                                            *args, **kwargs)

        if dropout_active and (self.dropout > 0.):
            retain_prob = 1 - self.dropout
            mask = layers.srng.binomial(input.shape, p=retain_prob,
                                        dtype='int32').astype('float32')
            # apply the input mask and rescale the input accordingly.
            # By doing this it's no longer necessary to rescale the weights
            # at test time.
            input = input / retain_prob * mask

        contiguous_input = gpu_contiguous(input)
        contiguous_filters = gpu_contiguous(self.W)
        if self.stride == 1:
            deconved = self.image_acts_op(contiguous_input, contiguous_filters)
        else:
            _, x, y, _ = self.get_output_shape()
            deconved = self.image_acts_op(contiguous_input, contiguous_filters,
                                          as_tensor_variable((x, y)))
        return self.nonlinearity(deconved)