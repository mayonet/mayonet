from __future__ import print_function
from pylearn2.models import mlp, maxout
from pylearn2.costs.mlp.dropout import Dropout
from pylearn2.space import Conv2DSpace
from pylearn2.training_algorithms import sgd, learning_rule
from pylearn2.termination_criteria import EpochCounter
from pylearn2.train import Train
from pylearn2.train_extensions import best_params, window_flip
import time

from mnist_dataset import MnistDataset
from dataset import load_pickle
import os
from pylearn2.utils import serial
from pylearn2.monitor import push_monitor


import sys
import rotator


class Logger(object):
    def __init__(self, filename="Default.log"):
        filename = os.path.join('logs', filename)
        self.terminal = sys.stdout
        self.log = open(filename, "a", 0)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(time.strftime("%Y.%m.%d %H:%M:%S --- ") + message)

    def __getattr__(self, attr):
        return getattr(self.terminal, attr)

for s in [80]:
    model_save_path = "last_model.pkl"
    watcher_save_path = "best_"+model_save_path

    # logger_name = model_save_path
    logger_name = time.strftime("%Y.%m.%d %H:%M:%S")
    if os.path.isfile(model_save_path):
        logger_name += "_c"
    else:
        logger_name += "_s"
    sys.stdout = Logger(logger_name + ".log")

    BATCH_SIZE = 64

    # trn = MnistDataset('train')
    # vld = MnistDataset('valid')
    # tst = MnistDataset('test')
    # nvis = 28*28

    # trn = PlanktonDataset(batch_size=BATCH_SIZE, which_set='train', one_hot=True)
    # vld = PlanktonDataset(batch_size=BATCH_SIZE, which_set='valid', one_hot=True)
    # vnis = 4096

    trn = load_pickle(BATCH_SIZE, 'train', method='bluntresize', size=s)
    vld = load_pickle(BATCH_SIZE, 'valid', method='bluntresize', size=s)

    window = (s-2, s-2)
    n_classes = trn.n_classes
    in_space = Conv2DSpace(shape=window,
                           num_channels=1,
                           axes=trn.view_converter.axes,
                           dtype='float32')

    if os.path.isfile(model_save_path):
        print('Loading model from %s' % model_save_path)
        mdl = push_monitor(serial.load(model_save_path), 'monitor')
    else:
        l0 = mlp.ConvRectifiedLinear(output_channels=36,
                                     kernel_shape=(3, 3),
                                     pool_shape=(2, 2),
                                     pool_stride=(2, 2),
                                     layer_name='l0',
                                     irange=0.05,
                                     max_kernel_norm=.9365)
        l1 = mlp.ConvRectifiedLinear(output_channels=64,
                                     kernel_shape=(3, 3),
                                     pool_shape=(2, 2),
                                     pool_stride=(2, 2),
                                     layer_name='l1',
                                     irange=0.05,
                                     max_kernel_norm=.9365)
        l2 = mlp.ConvRectifiedLinear(output_channels=64,
                                     kernel_shape=(3, 3),
                                     pool_shape=(2, 2),
                                     pool_stride=(2, 2),
                                     layer_name='l2',
                                     irange=0.05,
                                     max_kernel_norm=.9365)
        l3 = mlp.ConvRectifiedLinear(output_channels=128,
                                     kernel_shape=(3, 3),
                                     pool_shape=(2, 2),
                                     pool_stride=(2, 2),
                                     layer_name='l3',
                                     irange=0.05,
                                     max_kernel_norm=.9365)

        mo = maxout.Maxout(layer_name='mo',
                           irange=.005,
                           num_units=500,
                           num_pieces=5,
                           max_col_norm=.9)

        fc0 = mlp.RectifiedLinear(dim=1000,
                                  layer_name='fc0',
                                  irange=0.05,
                                  max_col_norm=.9,
                                  init_bias=0.01)

        fc1 = mlp.RectifiedLinear(dim=1000,
                                  layer_name='fc1',
                                  irange=0.05,
                                  max_col_norm=.9,
                                  init_bias=0.01)

        output = mlp.Softmax(layer_name='y',
                             n_classes=n_classes,
                             irange=0.1,
                             max_col_norm=1.9365)

        layers = [l0, l1, l2, l3, fc0, fc1, output]
        mdl = mlp.MLP(layers=layers,
                      input_space=in_space
                      )

    trainer = sgd.SGD(learning_rate=5e-2,
                      batch_size=BATCH_SIZE,
                      learning_rule=learning_rule.Momentum(.5),
                      cost=Dropout(default_input_scale=1.,
                                   default_input_include_prob=1.,
                                   input_include_probs={'fc0': 0.5, 'fc1': 0.5},
                                   input_scales={'fc0': 2., 'fc1': 2.}),
                      # termination_criterion=EpochCounter(max_epochs=275),
                      monitoring_dataset={'valid': vld,
                                          'train': trn})


    velocity = learning_rule.MomentumAdjustor(final_momentum=.65,
                                              start=1,
                                              saturate=250)

    watcher = best_params.MonitorBasedSaveBest(
        channel_name='valid_y_nll',
        save_path=watcher_save_path)

    decay = sgd.LinearDecayOverEpoch(start=1,
                                     saturate=250,
                                     decay_factor=.01)

    rtr = rotator.Rotator(window=window, randomize=[trn], center=[vld],
                          x_offsets=range(2),
                          y_offsets=range(2),
                          flip=True,
                          scales=[1.05**p for p in range(-8, 8)])

    experiment = Train(dataset=trn,
                       model=mdl,
                       algorithm=trainer,
                       extensions=[watcher, decay, velocity, rtr],
                       save_path=model_save_path,
                       save_freq=1)

    experiment.main_loop()
