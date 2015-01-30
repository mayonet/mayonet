#!/usr/bin/env python
from pylearn2.utils import serial
from theano import tensor as T
from theano import function
import numpy as np
import csv
from dataset import Dataset


# def process(mdl, ds, batch_size=100):
#     # This batch size must be evenly divisible into number of total samples!
#     mdl.set_batch_size(batch_size)
#     X = mdl.get_input_space().make_batch_theano()
#     Y = mdl.fprop(X)
#     y = T.argmax(Y, axis=1)
#     f = function([X], y)
#     yhat = []
#     for i in xrange(ds.X.shape[0] / batch_size):
#         x_arg = ds.X[i * batch_size:(i + 1) * batch_size, :]
#         x_arg = ds.get_topological_view(x_arg)
#         yhat.append(f(x_arg.astype(X.dtype)))
#     return np.array(yhat)


preprocessor = serial.load('/plankton/fixed64_preprocessor.pkl')
mdl = serial.load('conv_maxout1_zca.pkl')
fname = 'conv_maxout1_results.csv'

batch_size = 64
ds = Dataset('test', batch_size=batch_size)
ds.apply_preprocessor(preprocessor=preprocessor, can_fit=False)

X = mdl.get_input_space().make_batch_theano()
Y1 = mdl.fprop(X)
Y = T.clip(Y1, 0.0001, 0.9999)
f = function([X], Y)

yhat = []
for i in xrange(ds.X.shape[0] / batch_size):
    x_arg = ds.X[i * batch_size:(i + 1) * batch_size, :]
    x_arg = ds.get_topological_view(x_arg)
    yhat.append(f(x_arg.astype(X.dtype)))
y = np.vstack(yhat)


header = ['image'] + ds.label_names
with open(fname, 'w') as f:
    f.write(','.join(header) + '\n')
    for i, fn in enumerate(ds.test_fns):
        f.write(fn + ',' + ','.join('%.8f' % prob for prob in y[i]) + '\n')


# converted_results = [['id', 'label']] + [[n + 1, ds.unconvert(int(x))]
#                                          for n, x in enumerate(res.ravel())]
# with open(fname, 'w') as f:
#     csv_f = csv.writer(f, delimiter=',', quoting=csv.QUOTE_NONE)
#     csv_f.writerows(converted_results)
