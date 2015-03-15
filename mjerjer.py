from __future__ import print_function, division
import gzip
import pandas as pd
import numpy as np
import theano
import os
from np_dataset import load_npys
from ann import *

theano.config.floatX = 'float32'
theano.config.blas.ldflags = '-lblas -lgfortran'
floatX = theano.config.floatX


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


def prepare_features(img_names, csv_names, dtype='float32'):
    Xs = np.zeros((img_names.shape[0], 0), dtype=dtype)
    for csv in csv_names:
        with gzip.open(csv, 'rb') as f:
            table = pd.read_csv(f).set_index('image')
            Xs = np.hstack((Xs, table.loc[img_names].values.astype(dtype)))
    return Xs


csvs = ['/home/yoptar/git/subway-plankton/submissions/igipop_polished_40/results_all.csv.gz',
        '/home/yoptar/git/subway-plankton/submissions/amazon_train_2.polished/results_all.csv.gz',
        '/home/yoptar/git/subway-plankton/submissions/amazon_train1/results_all.csv.gz',
        '/home/yoptar/git/subway-plankton/submissions/amazon_train0/results_all.csv.gz',
        # '/home/yoptar/git/subway-plankton/normalized_properties.csv.gz'
        ]

_, train_y, train_names = load_npys('all', 1, 'crop')
_, valid_y, valid_names = load_npys('valid', 1, 'crop')

train_x = prepare_features(train_names, csvs, floatX)
valid_x = prepare_features(valid_names, csvs, floatX)

model_fn = 'mjerjer.pkl'

if os.path.isfile(model_fn) and True:
    logger.write('Loading model from %s...' % model_fn)
    mlp = cPickle.load(open(model_fn, 'rb'))
else:
    prelu_alpha = 0.25
    mlp = MLP([
        Dropout(0.8),
        DenseLayer(2000),
        PReLU(prelu_alpha),

        Dropout(0.5),
        DenseLayer(2000, leaky_relu_alpha=prelu_alpha),
        PReLU(prelu_alpha),

        # Dropout(0.5),
        # DenseLayer(2000, leaky_relu_alpha=prelu_alpha),
        # PReLU(prelu_alpha),

        # Dropout(0.5),
        # DenseLayer(2000, leaky_relu_alpha=prelu_alpha),
        # PReLU(prelu_alpha),

        Dropout(0.5),
        DenseLayer(train_y.shape[1], leaky_relu_alpha=prelu_alpha),
        NonLinearity(activation=T.nnet.softmax)
    ], [train_x.shape[1]],
        logger=logger)

l2 = 0
learning_rate = 1e-3  # np.exp(-2)
momentum = 0.99
epoch_count = 1000
batch_size = 100
minibatch_count = (train_y.shape[0]-1) // batch_size + 1
learning_decay = 0.5 ** (1./(200 * minibatch_count))
momentum_decay = 1  # 0.5 ** (1./(1000 * minibatch_count))
lr_min = 1e-15
mm_min = 0.5
valid_rnd_count = 1

method = 'nesterov'

logger.write('batch=%d, l2=%f, method=%s\nlr=%f, lr_decay=%f,\nmomentum=%f, momentum_decay=%f' %
             (batch_size, l2, method, learning_rate, learning_decay, momentum, momentum_decay))

tr = Trainer(mlp, batch_size, learning_rate, train_x, train_y, valid_X=valid_x, valid_y=valid_y, method=method,
             momentum=momentum, lr_decay=learning_decay, lr_min=lr_min, l2=l2, mm_decay=momentum_decay, mm_min=mm_min,
             model_file_name=model_fn, save_freq=1, save_in_different_files=True, epoch_count=epoch_count)

for res in tr():
    logger.write('{epoch}\t{train_nll:.5f}\t{test_nll:.5f}\t{epoch_time:.1f}s\t{valid_misclass:.2f}%\t{lr:.10f}\t'
                 '{momentum:.9f}\t{l2_error:.5f}'.format(**res))