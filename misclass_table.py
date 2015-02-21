from __future__ import print_function
from operator import itemgetter

import time
from pylearn2.utils import serial
from theano import function
import numpy as np
from dataset import load_pickle, PlanktonDataset
from matplotlib.pyplot import imshow, scatter

import rotator

from sklearn import manifold


def log(message):
    print((time.strftime("%Y.%m.%d %H:%M:%S --- ") + message))


model_path = "submissions/2015.02.14.0/validator_best_model.pkl"

BATCH_SIZE = 64

repetitions = 20

mdl = serial.load(model_path)
log('opened model')

# ds = load_pickle(BATCH_SIZE, which_set='test')
img_size = 80
window = (img_size-2, img_size-2)
ds = PlanktonDataset(BATCH_SIZE, 'valid', True, img_size, 'bluntresize')
log('created dataset')

y = None

y0 = np.zeros((ds.X.shape[0], 121))
rtr = rotator.Rotator(window, [ds], [],
                      x_offsets=range(2),
                      y_offsets=range(2),
                      median_radii=[0],
                      mean_radii=[0],
                      scales=[1.01**(p*abs(p)) for p in map(lambda x: x/2., range(-11, 9))],
                      flip=True)

for i in range(repetitions):
    rtr.randomize_datasets()
    log('rotated %i' % i)
    X = mdl.get_input_space().make_batch_theano()
    Y = mdl.fprop(X)
    f = function([X], Y)

    yhat = []
    for j in xrange(ds.X.shape[0] / BATCH_SIZE):
        x_arg = ds.X[j * BATCH_SIZE:(j + 1) * BATCH_SIZE, :]
        x_arg = ds.get_topological_view(x_arg)
        yhat.append(f(x_arg.astype(X.dtype)))
    y0 += np.vstack(yhat)



y0 /= repetitions
y0.clip(1e-12, 1 - 1e-12)
y0 = y0 / y0.sum(axis=1)[:, np.newaxis]

def nll(y0, y):
    return np.average(np.sum(-y*np.log(y0), axis=1))


def nll_class(y0, y, class_id):
    return nll(y0[y[:, class_id] == 1], y[y[:, class_id] == 1])

res = []
for class_id, class_label in enumerate(ds.unique_labels):
    k = y[:, class_id] == 1
    n = y[k].sum()
    L = nll(y0[k], y[k])
    res.append((class_id, class_label, L, n, n*L))

# sorted(res, key=itemgetter(4))

T = -(y*np.log(y0) + (1-y)*np.log(1-y0))
cl = np.zeros((ds.n_classes, ds.n_classes))
for class_id, class_label in enumerate(ds.unique_labels):
    k = y[:, class_id] == 1
    n = y[k].sum()
    cl[class_id] = np.sum(T[k], axis=0) / n

# imshow(cl)

similars = cl / cl.max()
similars = similars + (np.identity(len(similars)) - similars * np.identity(len(similars)))

# imshow(similars)

similars = (similars + similars.T)/2

mds = manifold.MDS(n_components=2, max_iter=3000, eps=1e-9, random_state=2005, dissimilarity="precomputed", n_jobs=1)
pos = mds.fit(similars).embedding_

scatter(pos[:, 0], pos[:, 1])