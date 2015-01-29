from __future__ import print_function
import os

from pylearn2.datasets import dense_design_matrix
from pylearn2.utils import string_utils
import numpy as np

from data_prep import resize_image
from skimage.io import imread


DATA_DIR = '/plankton'
TRAIN_DATA_DIR = os.path.join(DATA_DIR, 'train')
TEST_DATA_DIR = os.path.join(DATA_DIR, 'test')


def _read_labels():
    return os.listdir(TRAIN_DATA_DIR)


def _iterate_train_data_paths():
    for label in _read_labels():
        for pict in os.listdir(os.path.join(TRAIN_DATA_DIR, label)):
            yield os.path.join(TRAIN_DATA_DIR, label, pict), label


class Dataset(dense_design_matrix.DenseDesignMatrix):
    def __init__(self, which_set):
        # we define here:
        dtype = 'uint8'
        axes = ('b', 0, 1, 'c')
        ntrain = 50000
        nvalid = 0  # artefact, we won't use it
        ntest = 10000

        # we also expose the following details:
        self.img_shape = (1, 64, 64)
        self.img_size = np.prod(self.img_shape)
        self.label_names = _read_labels()
        self.label_to_int = {self.label_names[i]: i for i in range(len(self.label_names))}
        self.n_classes = len(self.label_names)

        fns, labels = zip(*_iterate_train_data_paths())

        lenx = len(labels)
        x = np.zeros((lenx, self.img_size), dtype=dtype)
        y = np.zeros((lenx, 1), dtype=dtype)

        for i in range(lenx):
            x[i] = resize_image(imread(fns[i], as_grey=True)).reshape(self.img_size)
            y[i] = self.label_to_int[labels[i]]



        view_converter = dense_design_matrix.DefaultViewConverter((64, 64, 1),
                                                                  axes)

        super(Dataset, self).__init__(X=np.cast['float32'](x), y=y, view_converter=view_converter,
                                      y_labels=self.n_classes)

Dataset('train')
