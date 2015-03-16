from __future__ import print_function, division
import gzip
import pandas as pd
import os
from dataset import read_labels
from np_dataset import load_npys
from ann import *

theano.config.floatX = 'float32'
theano.config.blas.ldflags = '-lblas -lgfortran'
floatX = theano.config.floatX

start_time = time.time()


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


def log(message):
    print((time.strftime("%Y.%m.%d %H:%M:%S --- ") + message))


def prepare_features(img_names, csv_names, dtype='float32'):
    Xs = np.zeros((img_names.shape[0], 0), dtype=dtype)
    for csv in csv_names:
        with gzip.open(csv, 'rb') as f:
            table = pd.read_csv(f).set_index('image')
            Xs = np.hstack((Xs, table.loc[img_names].values.astype(dtype)))
    return Xs

csvs = ['/home/yoptar/git/subway-plankton/submissions/igipop_polished_40/results.csv.gz',
        '/home/yoptar/git/subway-plankton/submissions/amazon_train_2.polished/results.csv.gz',
        '/home/yoptar/git/subway-plankton/submissions/amazon_train1/results.csv.gz',
        '/home/yoptar/git/subway-plankton/submissions/amazon_train0/results.csv.gz',
        ]

_, _, names = load_npys('test', 1, 'crop')

Xs = prepare_features(names, csvs, floatX)

model_fn = 'submissions/mjejer_gaussian/mjerjer.pkl'
res_name = 'submissions/mjejer_gaussian/results'

mdl = cPickle.load(open(model_fn, 'rb'))
log('opened model at ' + model_fn)

unique_labels = read_labels()

X = T.matrix('X', dtype=floatX)
Y = mdl.forward(X)
f = theano.function([X], Y)

batch_size = 100
batch_count = (Xs.shape[0]-1) // batch_size + 1

y = np.zeros((0, 121), floatX)
for i in range(batch_count):
    if (i-1) % 100 == 0:
        log('doing batch %i of %i' % (i-1, batch_count))
    y = np.vstack((y, f(Xs[i*batch_size:(i+1)*batch_size])))

header = ['image'] + unique_labels

with gzip.open(res_name + '.csv.gz', 'wb') as f:
    f.write(','.join(header) + '\n')
    for i, fn in enumerate(names):
        f.write(fn + ',' + ','.join('%.8f' % prob for prob in y[i]) + '\n')

log('Created "%s" from "%s" in %.1f minutes' % (res_name, model_fn, (time.time() - start_time)/60))