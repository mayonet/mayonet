from __future__ import print_function

import cPickle
import os
import theano
import theano.tensor as T
import numpy as np
import time
import gzip

from np_dataset import load_npys
from dataset import read_labels


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

theano.config.floatX = 'float32'
theano.config.blas.ldflags = '-lblas -lgfortran'
floatX = theano.config.floatX

start_time = time.time()

model_save_path = "junk/81_last_model_50_full.pkl"

res_name = 'results'

mdl = cPickle.load(open(model_save_path, 'rb'))
log('opened model at ' + model_save_path)

img_size = 78
max_offset = 0
window = (img_size-max_offset*2, img_size-max_offset*2)

Xs, _, names = load_npys('test', img_size)
Xs = Xs.reshape(Xs.shape[0], 1, np.sqrt(Xs.shape[1]), np.sqrt(Xs.shape[1]))
Xs = np.cast[floatX](1 - Xs/255.)

batch_size = 100
batch_count = (Xs.shape[0]-1) // batch_size + 1

unique_labels = read_labels()
n_classes = len(unique_labels)
label_to_int = {unique_labels[i]: i for i in range(n_classes)}

X = T.tensor4('X', dtype=floatX)
Y = mdl.forward(X)
f = theano.function([X], Y)

y = []
for i in xrange(batch_count):
    if i % 100 == 0:
        log('doing batch %i of %i' % (i, batch_count))
    y.append(f(Xs[i*batch_size:(i+1)*batch_size]))
y = np.vstack(y)

header = ['image'] + unique_labels

with gzip.open(res_name + '.csv.gz', 'wb') as f:
    f.write(','.join(header) + '\n')
    for i, fn in enumerate(names):
        f.write(fn + ',' + ','.join('%.8f' % prob for prob in y[i]) + '\n')

log('Created "%s" from "%s" in %.1f minutes' % (res_name, model_save_path, (time.time() - start_time)/60))