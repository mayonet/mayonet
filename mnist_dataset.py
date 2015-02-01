from __future__ import print_function

from pylearn2.datasets import dense_design_matrix
from pylearn2.space import Conv2DSpace

import cPickle


DATA_PATH='/plankton/pylearn_data/mnist.pkl'


class MnistDataset(dense_design_matrix.DenseDesignMatrix):
    def __init__(self, which='train'):
        switch = {'train': 0, 'valid': 1, 'test': 2}
        data = cPickle.load(open(DATA_PATH))[switch[which]]
        self.n_classes = 10
        super(MnistDataset, self).__init__(X=data[0], y=data[1], y_labels=self.n_classes,)
        self.convert_to_one_hot()
        self.in_space = Conv2DSpace(shape=(28, 28),
                                    num_channels=1,
                                    axes=('b', 0, 1, 'c'))