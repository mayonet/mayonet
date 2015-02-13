from __future__ import print_function

import time
from pylearn2.utils import serial
from theano import function
import numpy as np
from dataset import load_pickle, PlanktonDataset
import gzip

import rotator


def log(message):
    print((time.strftime("%Y.%m.%d %H:%M:%S --- ") + message))

start_time = time.time()

model_save_path = "sub_last_model.pkl"

BATCH_SIZE = 32
subsets = 4
repetitions = 20

fname = 'results'

mdl = serial.load(model_save_path)
log('opened model')

# ds = load_pickle(BATCH_SIZE, which_set='test')
img_size = 80
window = (img_size-2, img_size-2)
ds = PlanktonDataset(BATCH_SIZE*subsets, 'test', False, img_size, 'bluntresize')
log('created dataset')

row_count = ds.X.shape[0]
backed = ds.get_topological_view()
step = row_count//subsets

y = None

for k in range(subsets):
    ds.set_topological_view(backed[step*k:step*(k+1), :, :, :], ds.view_converter.axes)
    y0 = np.zeros((ds.X.shape[0], 121))
    rtr = rotator.Rotator(window, [ds], [],
                          x_offsets=range(2),
                          y_offsets=range(2),
                          scales=[1.01**(p*abs(p)) for p in map(lambda x: x/2., range(-11, 9))],
                          flip=True)

    for i in range(repetitions):
        rtr.randomize_datasets()
        log('rotated %i in part %i' % (i, k))
        X = mdl.get_input_space().make_batch_theano()
        Y = mdl.fprop(X)
        f = function([X], Y)

        yhat = []
        for j in xrange(ds.X.shape[0] / BATCH_SIZE):
            x_arg = ds.X[j * BATCH_SIZE:(j + 1) * BATCH_SIZE, :]
            x_arg = ds.get_topological_view(x_arg)
            yhat.append(f(x_arg.astype(X.dtype)))
        y0 += np.vstack(yhat)

    y = np.vstack((y, y0)) if y is not None else y0

y /= repetitions

header = ['image'] + ds.unique_labels

log('creating csv-file')

with gzip.open(fname + '.csv.gz', 'wb') as f:
    f.write(','.join(header) + '\n')
    for i, fn in enumerate(ds.test_fns):
        f.write(fn + ',' + ','.join('%.8f' % prob for prob in y[i]) + '\n')

log('Created "%s" from "%s" in %.1f minutes' % (fname, model_save_path, (time.time() - start_time)/60))