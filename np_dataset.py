from __future__ import print_function
import os
import random
import numpy as np
from data_prep import read_image
from sklearn.cross_validation import StratifiedKFold

from dataset import *


def one_hot(y):
    n_classes = np.max(y)+1
    y0 = np.zeros((y.shape[0], n_classes), dtype='uint8')
    y0[np.arange(y.shape[0]), y] = 1
    return y0


def read_data(which_set='train', image_size=CROP_SIZE, resizing_method='bluntresize', only_labels=None):
    unique_labels = read_labels()
    n_classes = len(unique_labels)
    label_to_int = {unique_labels[i]: i for i in range(n_classes)}

    if which_set == 'test':
        test_fns = get_test_data_paths()
        x = np.zeros((len(test_fns), image_size*image_size), dtype='float32')
        for i, pic in enumerate(test_fns):
            x[i] = read_image(os.path.join(TEST_DATA_DIR, pic), image_size, resizing_method)
        y = None
    else:
        d = list(iterate_train_data_paths())
        random.seed(11)
        random.shuffle(d)
        fns, labels = zip(*d)

        n = len(fns)
        x = np.zeros((n, image_size*image_size), dtype='float32')
        y = np.zeros(n, dtype='uint8')

        for i in range(n):
            x[i] = read_image(fns[i], image_size, resizing_method)
            y[i] = label_to_int[labels[i]]

        for train_i, valid_i in StratifiedKFold(labels, 6):

            Xs = {'train': x[train_i],
                  'valid': x[valid_i],
                  'all': x}

            Ys = {'train': y[train_i],
                  'valid': y[valid_i],
                  'all': y}
            x = Xs[which_set]
            y = one_hot(Ys[which_set])
            break

    return x, y


def np_fn(data_dir, which_set, image_size, resizing_method, postfix='x'):
    return '%s/%s_%s%d_%s.npy' % (data_dir, which_set, resizing_method, image_size, postfix)


def create_npys(which_set='train', image_size=CROP_SIZE, resizing_method='bluntresize'):
    x, y = read_data(which_set, image_size, resizing_method)
    np.save(np_fn(DATA_DIR, which_set, image_size, resizing_method, 'x'), x)
    if y is not None:
        np.save(np_fn(DATA_DIR, which_set, image_size, resizing_method, 'y'), y)


def load_npys(which_set='train', image_size=CROP_SIZE, resizing_method='bluntresize'):
    assert which_set in ['train', 'test', 'valid']

    x_fn = np_fn(DATA_DIR, which_set, image_size, resizing_method, 'x')
    if not os.path.isfile(x_fn):
        print("File %s not found. Creating new..." % x_fn)
        create_npys(which_set, image_size, resizing_method)
    x = np.load(x_fn)
    if which_set == 'test':
        y = None
    else:
        y = np.load(np_fn(DATA_DIR, which_set, image_size, resizing_method, 'y'))
    return x, y


if __name__ == '__main__':
    print('Creating train npys...')
    create_npys('train', 98, 'bluntresize')
    print('Creating valid npys...')
    create_npys('valid', 98, 'bluntresize')
    print('Creating test npys...')
    create_npys('test', 98, 'bluntresize')