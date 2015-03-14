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


def read_data(which_set='train', image_size=CROP_SIZE, resizing_method='bluntresize', only_labels=None, seed=11):
    unique_labels = read_labels()
    n_classes = len(unique_labels)
    label_to_int = {unique_labels[i]: i for i in range(n_classes)}

    if which_set == 'test':
        test_fns = get_test_data_paths()
        x = np.zeros((len(test_fns), image_size*image_size), dtype='float32')
        for i, pic in enumerate(test_fns):
            x[i] = read_image(os.path.join(TEST_DATA_DIR, pic), image_size, resizing_method)
        y = None
        names = test_fns
    else:
        d = list(iterate_train_data_paths())
        random.seed(seed)
        random.shuffle(d)
        fns, labels = zip(*d)

        n = len(fns)
        x = np.zeros((n, image_size*image_size), dtype='float32')
        y = np.zeros(n, dtype='uint8')
        names = np.zeros(n, dtype='O')

        for i in range(n):
            x[i] = read_image(fns[i], image_size, resizing_method)
            y[i] = label_to_int[labels[i]]
            names[i] = os.path.basename(fns[i])

        for train_i, valid_i in StratifiedKFold(labels, 6):

            Xs = {'train': x[train_i],
                  'valid': x[valid_i],
                  'all': x}

            Ys = {'train': y[train_i],
                  'valid': y[valid_i],
                  'all': y}

            Names = {'train': names[train_i],
                     'valid': names[valid_i],
                     'all': names}
            x = Xs[which_set]
            y = one_hot(Ys[which_set])
            names = Names[which_set]
            break

    return x, y, names


def np_fn(data_dir, which_set, image_size, resizing_method, seed, postfix='x'):
    return '%s/%s_%s%d_%s_seed%d.npy' % (data_dir, which_set, resizing_method, image_size, postfix, seed)


def create_npys(which_set='train', image_size=CROP_SIZE, resizing_method='bluntresize', seed=11):
    x, y, names = read_data(which_set, image_size, resizing_method, seed)
    np.save(np_fn(DATA_DIR, which_set, image_size, resizing_method, seed, 'x'), x)
    np.save(np_fn(DATA_DIR, which_set, image_size, resizing_method, seed, 'names'), names)
    if y is not None:
        np.save(np_fn(DATA_DIR, which_set, image_size, resizing_method, seed, 'y'), y)


def load_npys(which_set='train', image_size=CROP_SIZE, resizing_method='bluntresize', seed=11):
    assert which_set in ['train', 'test', 'valid']

    x_fn = np_fn(DATA_DIR, which_set, image_size, resizing_method, seed, 'x')
    if not os.path.isfile(x_fn):
        print("File %s not found. Creating new..." % x_fn)
        create_npys(which_set, image_size, resizing_method, seed)
    x = np.load(x_fn)
    names = np.load(np_fn(DATA_DIR, which_set, image_size, resizing_method, seed, 'names'))
    if which_set == 'test':
        y = None
    else:
        y = np.load(np_fn(DATA_DIR, which_set, image_size, resizing_method, seed, 'y'))
    return x, y, names


if __name__ == '__main__':
    print('Creating train npys...')
    create_npys('train', 98, 'bluntresize')
    print('Creating valid npys...')
    create_npys('valid', 98, 'bluntresize')
    print('Creating test npys...')
    create_npys('test', 98, 'bluntresize')