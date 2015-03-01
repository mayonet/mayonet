from __future__ import print_function
import os
import random

from pylearn2.datasets import dense_design_matrix
from pylearn2.datasets.preprocessing import Pipeline, GlobalContrastNormalization, ZCA
import numpy as np

from data_prep import read_image
from sklearn.cross_validation import StratifiedKFold
import doctest
from pylearn2.utils import serial


DATA_DIR = '/plankton'
TRAIN_DATA_DIR = os.path.join(DATA_DIR, 'train')
TEST_DATA_DIR = os.path.join(DATA_DIR, 'test')

CROP_SIZE = 98


def read_labels():
    return sorted(os.listdir(TRAIN_DATA_DIR))


def get_test_data_paths():
    return sorted(os.listdir(TEST_DATA_DIR))


def iterate_train_data_paths():
    for label in read_labels():
        for pict in os.listdir(os.path.join(TRAIN_DATA_DIR, label)):
            yield os.path.join(TRAIN_DATA_DIR, label, pict), label


def make_divisible_by_batch(indexi, batch_size):
    """
    indexi is ndarray
    >>> make_divisible_by_batch(np.arange(10), 6)
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1])
    """
    n = len(indexi)
    to_add = (batch_size - (n % batch_size)) % batch_size
    return np.concatenate((indexi, indexi[:to_add]))


def make_pickles(batch_size, method='crop', size=CROP_SIZE):

    preprocessor = Pipeline([#GlobalContrastNormalization(scale=55.)
                             #ZCA()
                             ])

    print("preprocessors - none")

    trn = PlanktonDataset(batch_size, 'train', image_size=size, resizing_method=method)
    trn.apply_preprocessor(preprocessor, True)
    serial.save(_pickle_fn(DATA_DIR, 'train', method, size), trn)

    vld = PlanktonDataset(batch_size, 'valid', image_size=size, resizing_method=method)
    vld.apply_preprocessor(preprocessor, False)
    serial.save(_pickle_fn(DATA_DIR, 'valid', method, size), vld)

    # tst = PlanktonDataset(batch_size, 'test', one_hot=False, method=method)
    # tst.apply_preprocessor(preprocessor, False)
    # serial.save(_pickle_fn(DATA_DIR, 'test', method, size), tst)

    print("made pickles")


def drop_pickles():
    """Depricated"""
    for which in ['train', 'test', 'valid']:

        fn = os.path.join(DATA_DIR, which + '.pkl')
        try:
            os.remove(fn)
        except OSError:
            print('"%s" not found' % fn)


def _pickle_fn(data_path, which_set, method, size):
    return os.path.join(data_path, '%s_%s%d.pkl' % (which_set, method, size))


def load_pickle(batch_size, which_set='train', method='crop', size=CROP_SIZE):
    assert which_set in ['train', 'test', 'valid']

    path = _pickle_fn(DATA_DIR, which_set, method, size)

    if not os.path.isfile(path):
        make_pickles(batch_size, method, size)

    return serial.load(path)


class PlanktonDataset(dense_design_matrix.DenseDesignMatrix):

    def __init__(self, batch_size, which_set='train', one_hot=True, image_size=CROP_SIZE, resizing_method='crop'):

        self.unique_labels = read_labels()
        self.n_classes = len(self.unique_labels)
        self.label_to_int = {self.unique_labels[i]: i for i in range(self.n_classes)}

        d = list(iterate_train_data_paths())
        random.seed(11)
        random.shuffle(d)
        fns, labels = zip(*d)

        n = len(fns)
        x = np.zeros((n, image_size*image_size), dtype='float32')
        y = np.zeros(n, dtype='uint8')

        for i in range(n):
            x[i] = read_image(fns[i], image_size, resizing_method)
            y[i] = self.label_to_int[labels[i]]

        for train_i, valid_i in StratifiedKFold(labels, 6):
            train_i = make_divisible_by_batch(train_i, batch_size)
            valid_i = make_divisible_by_batch(valid_i, batch_size)
            if which_set == 'test':
                self.test_fns = get_test_data_paths()
                test_X = np.zeros((len(self.test_fns), image_size*image_size), dtype='float32')
                for i, pic in enumerate(self.test_fns):
                    test_X[i] = read_image(os.path.join(TEST_DATA_DIR, pic), image_size, resizing_method)
                test_i = make_divisible_by_batch(np.arange(test_X.shape[0]), batch_size)
            else:
                test_X= np.zeros(5)
                test_i = range(3)

            Xs = {'train': x[train_i],
                  'valid': x[valid_i],
                  'all': x,
                  'test': test_X[test_i]}

            Ys = {'train': y[train_i],
                  'valid': y[valid_i],
                  'all': y,
                  'test': None}
            break

        x = ((255 - Xs[which_set]) / 255.)
        y = Ys[which_set]
        y_labels = self.n_classes

        if which_set == 'test':
            y_labels = None

        axes = ('b', 0, 1, 'c')

        view_converter = dense_design_matrix.DefaultViewConverter((image_size, image_size, 1), axes)

        super(PlanktonDataset, self).__init__(X=x, y=y, y_labels=y_labels, view_converter=view_converter)
        if one_hot:
            self.convert_to_one_hot()
            delattr(self, "y_labels")


if __name__ == '__main__':
    doctest.testmod()