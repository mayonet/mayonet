from __future__ import print_function

from time import time
from pylearn2.utils import serial
from theano import function
import numpy as np
from dataset import load_pickle, PlanktonDataset
import gzip

start_time = time()

model_save_path = "best_last_model.pkl"

BATCH_SIZE = 32

fname = 'results.csv.gz'

mdl = serial.load(model_save_path)

# ds = load_pickle(BATCH_SIZE, which_set='test')
ds = PlanktonDataset(BATCH_SIZE, 'test', False, 64)

X = mdl.get_input_space().make_batch_theano()
Y = mdl.fprop(X)
f = function([X], Y)

yhat = []
for i in xrange(ds.X.shape[0] / BATCH_SIZE):
    x_arg = ds.X[i * BATCH_SIZE:(i + 1) * BATCH_SIZE, :]
    x_arg = ds.get_topological_view(x_arg)
    yhat.append(f(x_arg.astype(X.dtype)))
y = np.vstack(yhat)


header = ['image'] + ds.unique_labels
with gzip.open(fname, 'wb') as f:
    f.write(','.join(header) + '\n')
    for i, fn in enumerate(ds.test_fns):
        f.write(fn + ',' + ','.join('%.8f' % prob for prob in y[i]) + '\n')
    f.close()

print('Created "%s" from "%s" in %.1f minutes' % (fname, model_save_path, (time() - start_time)/60))