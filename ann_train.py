#!/usr/bin/env python2
from __future__ import division, print_function
from operator import itemgetter
import os
from pprint import pprint
from dataset import read_labels
from hierarchy import heated_targetings
from np_dataset import load_npys
import theano
# theano.config.compute_test_value = 'warn'
from rotator import randomize_dataset_bc01

# theano.config.exception_verbosity = 'high'
theano.config.floatX = 'float32'
theano.config.blas.ldflags = '-lblas -lgfortran'
floatX = theano.config.floatX
from ann import *


class Logger(object):
    def __init__(self, filename="Default.log"):
        filename = os.path.join('logs', filename)
        self.log = open(filename, "a", 0)

    def write(self, message):
        if message != '\n':
            print(message)
            self.log.write(time.strftime("%Y.%m.%d %H:%M:%S --- ") + message + "\n")

    def close(self):
        self.log.close()

logger_name = time.strftime("%Y-%m-%dT%H-%M-%S.log")
logger = Logger(logger_name)


np.random.seed(1100)

dtype_y = 'uint8'
model_fn = 'last_model_80_partial.pkl'

unique_labels = read_labels()
n_classes = len(unique_labels)
label_to_int = {unique_labels[i]: i for i in range(n_classes)}
new_label_to_int = {}
siphonophore_columns = []
i = 0
for lbl, idx in sorted(label_to_int.items(), key=itemgetter(1)):
    if lbl.startswith('siphonophore') or lbl.startswith('acantharia') or lbl.startswith('unknown'):
        siphonophore_columns.append(idx)
        new_label_to_int[lbl] = i
        print(lbl)
        i += 1
siphonophore_columns = np.array(siphonophore_columns)
label_to_int = new_label_to_int

img_size = 80
max_offset = 1
window = (img_size-2*max_offset, img_size-2*max_offset)

train_x, train_y, _ = load_npys(which_set='train', image_size=img_size)
valid_x, valid_y, _ = load_npys(which_set='valid', image_size=img_size)

train_y = train_y[:, siphonophore_columns]
train_x = train_x[np.max(train_y, axis=1) == 1, :]
train_y = train_y[np.max(train_y, axis=1) == 1, :]
valid_y = valid_y[:, siphonophore_columns]
valid_x = valid_x[np.max(valid_y, axis=1) == 1, :]
valid_y = valid_y[np.max(valid_y, axis=1) == 1, :]

train_x = train_x.reshape((train_x.shape[0], 1, np.sqrt(train_x.shape[1]), np.sqrt(train_x.shape[1])))
valid_x = valid_x.reshape((valid_x.shape[0], 1, np.sqrt(valid_x.shape[1]), np.sqrt(valid_x.shape[1])))
train_x = np.cast[floatX](1 - train_x/255.)
valid_x = np.cast[floatX](1 - valid_x/255.)

cost = neg_log_likelihood
valid_cost = cost

# cost = soft_log_likelihood


def heat_ys(y):
    cl = zip(0.7**np.arange(1, 10), np.exp(np.arange(2, 11)))
    heat_ys.iter = min(heat_ys.iter, len(cl)-1)
    res = heated_targetings(label_to_int, y,
                            cl[heat_ys.iter][0], cl[heat_ys.iter][0], cl[heat_ys.iter][1])
    heat_ys.counter += 1
    if heat_ys.counter % 3 == 0:
        heat_ys.iter += 1
    return res
heat_ys.counter = 0
heat_ys.iter = 0

len_in = train_x.shape[1]
len_out = train_y.shape[1]

if os.path.isfile(model_fn) and False:
    logger.write('Loading model from %s...' % model_fn)
    mlp = cPickle.load(open(model_fn, 'rb'))
else:
    prelu_alpha = 0.25
    mlp = MLP([
        GaussianDropout(0.03),
        ConvolutionalLayer((3, 3), 64, leaky_relu_alpha=1),
        MaxPool((2, 2)),
        PReLU(prelu_alpha),

        GaussianDropout(0.03),
        ConvolutionalLayer((3, 3), 128, pad=1, leaky_relu_alpha=prelu_alpha),
        # MaxPool((2, 2)),
        PReLU(prelu_alpha),

        GaussianDropout(0.03),
        ConvolutionalLayer((3, 3), 128, leaky_relu_alpha=prelu_alpha),
        MaxPool((2, 2)),
        NonLinearity(),

        GaussianDropout(0.03),
        ConvolutionalLayer((3, 3), 256, pad=1),
        # MaxPool((2, 2)),
        NonLinearity(),

        GaussianDropout(0.03),
        ConvolutionalLayer((3, 3), 256, pad=1),
        # MaxPool((2, 2)),
        NonLinearity(),

        GaussianDropout(0.03),
        ConvolutionalLayer((3, 3), 256),
        MaxPool((2, 2)),
        NonLinearity(),

        GaussianDropout(0.03),
        ConvolutionalLayer((3, 3), 256, pad=1),
        # MaxPool((2, 2)),
        NonLinearity(),

        GaussianDropout(0.03),
        ConvolutionalLayer((3, 3), 256, pad=1),
        # MaxPool((2, 2)),
        NonLinearity(),

        GaussianDropout(0.03),
        ConvolutionalLayer((3, 3), 256, pad=1),
        # MaxPool((2, 2)),
        NonLinearity(),

        GaussianDropout(0.03),
        ConvolutionalLayer((3, 3), 256),
        MaxPool((2, 2)),
        NonLinearity(),

        Flatten(),

        # GaussianDropout(1),
        DenseLayer(2000),
        NonLinearity(),
        # PReLU(prelu_alpha),
        # Maxout(pieces=4),

        # GaussianDropout(1),
        DenseLayer(2000),
        NonLinearity(),
        # PReLU(prelu_alpha),
        # Maxout(pieces=4),

        # GaussianDropout(1),
        DenseLayer(len_out),
        NonLinearity(activation=T.nnet.softmax)
    ], (1,) + window, logger)



## TODO move to mlp.get_updates
l2 = 0  # 1e-5
learning_rate = 1  # np.exp(-2)
momentum = 0.95
epoch_count = 1000
batch_size = 64
minibatch_count = (train_x.shape[0]-1) // batch_size + 1
learning_decay = 1  # 0.5 ** (1./(250 * minibatch_count))
momentum_decay = 1  # 0.5 ** (1./(1000 * minibatch_count))
lr_min = 1e-15
mm_min = 0.5
valid_rnd_count = 10

method = 'adadelta'

logger.write('batch=%d, l2=%f, method=%s\nlr=%f, lr_decay=%f,\nmomentum=%f, momentum_decay=%f' %
             (batch_size, l2, method, learning_rate, learning_decay, momentum, momentum_decay))

scales = [1.01**(p*abs(p)) for p in map(lambda x: x/2., range(-7, 11))]

randomization_params = {
    'window': window,
    'scales': scales,
    'angles': range(360),
    'x_offsets': range(max_offset+1),
    'y_offsets': range(max_offset+1),
    # 'median_radii': [0, 0, 0, 1],
    # 'mean_radii': [0, 0, 0, 0, 1, 1, 2],
    'flip': True
}
print(randomization_params, file=logger)


def randomize(dataset):
    return map(lambda ds: randomize_dataset_bc01(ds, **randomization_params) if ds.ndim == 4 else ds, dataset)

tr = Trainer(mlp, batch_size, learning_rate, train_x, train_y, valid_X=valid_x, valid_y=valid_y, method=method,
             momentum=momentum, lr_decay=learning_decay, lr_min=lr_min, l2=l2, mm_decay=momentum_decay, mm_min=mm_min,
             train_augmentation=randomize, valid_augmentation=randomize, valid_aug_count=valid_rnd_count,
             # train_y_augmentation=heat_ys,
             model_file_name=model_fn, save_freq=1, save_in_different_files=True, epoch_count=epoch_count,
             cost_f=cost, valid_cost_f=valid_cost)
with open('monitors_%s.csv' % model_fn, 'w', 0) as monitor_csv:
    logger.write('#\tt_nll\tv_nll\ttime\trclass\tlearning rate\tmomentum\tl2_error')  # header
    for res in tr():
        logger.write('{epoch}\t{train_nll:.5f}\t{test_nll:.5f}\t{epoch_time:.1f}s\t{valid_misclass:.2f}%\t{lr:.10f}\t'
                     '{momentum:.9f}\t{l2_error:.5f}'.format(**res))
        if res['epoch'] == 0:
            print('\t'.join(res), file=monitor_csv)
        print('\t'.join([str(v) for v in res.itervalues()]), file=monitor_csv)