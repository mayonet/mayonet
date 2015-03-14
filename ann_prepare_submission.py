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
from rotator import randomize_dataset_bc01


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

model_save_path = "submissions/amazon_train0/35_amazon_train_0.pkl"

res_name = 'submissions/amazon_train0/results'

mdl = cPickle.load(open(model_save_path, 'rb'))
log('opened model at ' + model_save_path)

img_size = 100
max_offset = 1
window = (img_size-max_offset*2, img_size-max_offset*2)

Xs, _, names = load_npys('test', img_size, resizing_method='crop')
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

scales = [1.01**(p*abs(p)) for p in map(lambda x: x/2., range(-7, 11))]

randomization_params = {
    'window': window,
    'scales': scales,
    'angles': range(360),
    'x_offsets': range(max_offset+1),
    'y_offsets': range(max_offset+1),
    'flip': True
}

polish_randomization_params = {
    'window': window,
    'scales': (1,),
    'angles': (0, 90),
    'x_offsets': range(max_offset+1),
    'y_offsets': range(max_offset+1),
    'flip': True
}


def polish_randomize(dataset):
    return randomize_dataset_bc01(dataset, **polish_randomization_params)


def randomize(dataset):
    return randomize_dataset_bc01(dataset, **randomization_params)

rotation_count = 10

y = []
for i in xrange(batch_count):
    if i % 100 == 0:
        log('doing batch %i of %i' % (i, batch_count))
    rXs = Xs[i*batch_size:(i+1)*batch_size]
    y0 = np.zeros((rXs.shape[0], 121), dtype=floatX)
    for r in xrange(rotation_count):
        y0 += f(polish_randomize(rXs))
    y.append(y0/rotation_count)
y = np.vstack(y)

header = ['image'] + unique_labels

with gzip.open(res_name + '.csv.gz', 'wb') as f:
    f.write(','.join(header) + '\n')
    for i, fn in enumerate(names):
        f.write(fn + ',' + ','.join('%.8f' % prob for prob in y[i]) + '\n')

log('Created "%s" from "%s" in %.1f minutes' % (res_name, model_save_path, (time.time() - start_time)/60))