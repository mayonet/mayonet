from __future__ import print_function
import os
import random

from pylearn2.datasets import dense_design_matrix
from pylearn2.utils import string_utils
import numpy as np

from data_prep import resize_image
from skimage.io import imread
from sklearn.cross_validation import StratifiedKFold
import doctest


DATA_DIR = '/plankton'
TRAIN_DATA_DIR = os.path.join(DATA_DIR, 'train')
TEST_DATA_DIR = os.path.join(DATA_DIR, 'test')


def _read_labels():
    return os.listdir(TRAIN_DATA_DIR)


def _iterate_train_data_paths():
    for label in _read_labels():
        for pict in os.listdir(os.path.join(TRAIN_DATA_DIR, label)):
            yield os.path.join(TRAIN_DATA_DIR, label, pict), label


def _make_dividible_by_batch(indexi, batch_size):
    """
    indexi is ndarray
    >>> _make_dividible_by_batch(np.arange(10), 6)
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1])
    """
    n = len(indexi)
    to_add = batch_size - (n % batch_size)
    return np.concatenate((indexi, indexi[:to_add]))


class Dataset(dense_design_matrix.DenseDesignMatrix):
    def __init__(self, which_set='train', batch_size=100, partial=0):
        # we define here:
        dtype = 'uint8'
        axes = ('b', 0, 1, 'c')

        # we also expose the following details:
        self.img_shape = (64, 64, 1)
        self.img_size = np.prod(self.img_shape)
        self.label_names = _read_labels()
        self.label_to_int = {self.label_names[i]: i for i in range(len(self.label_names))}
        self.n_classes = len(self.label_names)

        d = list(_iterate_train_data_paths())
        if partial == 1:
            d = d[:2000]
        random.seed(20015)
        random.shuffle(d)
        fns, labels = zip(*d)



        lenx = len(labels)
        # x = np.zeros((lenx, self.img_size), dtype=dtype)
        x = np.zeros((lenx,) + self.img_shape, dtype=dtype)
        #y = np.zeros((lenx, self.n_classes), dtype=dtype)
        y = np.zeros((lenx, 1), dtype=dtype)

        for i in range(lenx):
            x[i] = resize_image(imread(fns[i], as_grey=True))[:, :, np.newaxis]#.reshape(self.img_size)
            #y[i, self.label_to_int[labels[i]]] = 1
            y[i] = self.label_to_int[labels[i]]

        for train_i, valid_i in StratifiedKFold(labels, 5):
            train_i = _make_dividible_by_batch(train_i, batch_size)
            valid_i = _make_dividible_by_batch(valid_i, batch_size)
            Xs = {'train': x[train_i],
                  'valid': x[valid_i]}

            Ys = {'train': y[train_i],
                  'valid': y[valid_i]}
            break



        x = np.cast['float32'](Xs[which_set])
        y = Ys[which_set]



        view_converter = dense_design_matrix.DefaultViewConverter((64, 64, 1), axes)

        #super(Dataset, self).__init__(X=x, y=y, view_converter=view_converter, y_labels=self.n_classes)

        super(Dataset, self).__init__(topo_view=x, y=y, y_labels=self.n_classes)


if __name__ == '__main__':
    doctest.testmod()