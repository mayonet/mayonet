from __future__ import print_function
from pylearn2.models import mlp, maxout
from pylearn2.costs.mlp.dropout import Dropout
from pylearn2.space import Conv2DSpace
from pylearn2.training_algorithms import sgd, learning_rule
from pylearn2.termination_criteria import EpochCounter
from pylearn2.train import Train
from pylearn2.train_extensions import best_params, window_flip
import time
from itertools import product

from mnist_dataset import MnistDataset
from dataset import load_pickle
import os
from pylearn2.utils import serial
from pylearn2.monitor import push_monitor


import sys
import rotator
import validator


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

    def close(self):
        self.log.close()
        sys.stdout = self.terminal


# image size
s = 98
BATCH_SIZE = 100

med_r = [[0], [3]*1 + [2]*2 + [1]*4 + [0]*8]
mea_r = [[0], [2]*1 + [1]*2 + [0]*4]
mea = mea_r[0]
med = med_r[0]

exp_names = ['pylearn_vs_pycton']  # , 'maxout500', 'fc1000', 'maxout1000']

for exp_name in exp_names:

    # for iteration_num, (med, mea) in enumerate(product(med_r, mea_r)):
    model_save_path = "last_model_%s.pkl" % exp_name
    watcher_save_path = "best_"+model_save_path

    logger_name = model_save_path
    # logger_name = time.strftime("%Y-%m-%dT%H-%M-%S")
    if os.path.isfile(model_save_path):
        logger_name += "_c"
    else:
        logger_name += "_s"
    logger = Logger(logger_name + ".log")
    sys.stdout = logger

    # trn = MnistDataset('train')
    # vld = MnistDataset('valid')
    # tst = MnistDataset('test')
    # nvis = 28*28

    # trn = PlanktonDataset(batch_size=BATCH_SIZE, which_set='train', one_hot=True)
    # vld = PlanktonDataset(batch_size=BATCH_SIZE, which_set='valid', one_hot=True)
    # vnis = 4096

    trn = load_pickle(BATCH_SIZE, 'train', method='bluntresize', size=s)
    vld = load_pickle(BATCH_SIZE, 'valid', method='bluntresize', size=s)

    window = (s, s)
    n_classes = trn.n_classes
    in_space = Conv2DSpace(shape=window,
                           num_channels=1,
                           axes=trn.view_converter.axes,
                           dtype='float32')

    if os.path.isfile(model_save_path):
        print('Loading model from %s' % model_save_path)
        mdl = push_monitor(serial.load(model_save_path), 'monitor')
        # mdl = serial.load(model_save_path)
    else:
        c16 = mlp.ConvRectifiedLinear(output_channels=16,
                                      kernel_shape=(3, 3),
                                      pool_shape=(3, 3),
                                      pool_stride=(3, 3),
                                      layer_name='c16',
                                      irange=0.05)
        c32 = mlp.ConvRectifiedLinear(output_channels=32,
                                      kernel_shape=(3, 3),
                                      pool_shape=(3, 3),
                                      pool_stride=(3, 3),
                                      layer_name='c32',
                                      irange=0.05)
        c64 = mlp.ConvRectifiedLinear(output_channels=64,
                                      kernel_shape=(3, 3),
                                      pool_shape=(2, 2),
                                      pool_stride=(2, 2),
                                      layer_name='c64',
                                      irange=0.05)
        c6_ = mlp.ConvRectifiedLinear(output_channels=64,
                                      kernel_shape=(3, 3),
                                      pool_shape=(2, 2),
                                      pool_stride=(2, 2),
                                      layer_name='c64',
                                      irange=0.05)

        mo500 = maxout.Maxout(layer_name='maxout500',
                              irange=.005,
                              num_units=500,
                              num_pieces=5,
                              max_col_norm=.9)

        mo1000 = maxout.Maxout(layer_name='maxout1000',
                               irange=.005,
                               num_units=1000,
                               num_pieces=5,
                               max_col_norm=.9)

        fc0 = mlp.RectifiedLinear(dim=1200,
                                  layer_name='fc1200_0',
                                  irange=0.05,
                                  max_col_norm=1.9365,
                                  init_bias=0)

        fc1 = mlp.RectifiedLinear(dim=1200,
                                  layer_name='fc1200_1',
                                  irange=0.05,
                                  max_col_norm=1.9365,
                                  init_bias=0)

        fc5000 = mlp.RectifiedLinear(dim=5000,
                                     layer_name='fc5000',
                                     irange=0.05,
                                     max_col_norm=.9,
                                     init_bias=0.01)

        # exp_layer = {L.layer_name: L for L in [mo500, mo1000, fc1000, fc2500, fc5000]}

        # mo1 = maxout.Maxout(layer_name='mo1',
        #                     irange=.005,
        #                     num_units=500,
        #                     num_pieces=5,
        #                     max_col_norm=.9)

        # fc0 = mlp.RectifiedLinear(dim=1000,
        #                           layer_name='fc0',
        #                           irange=0.05,
        #                           max_col_norm=.9,
        #                           init_bias=0.01)
        #
        # fc1 = mlp.RectifiedLinear(dim=1000,
        #                           layer_name='fc1',
        #                           irange=0.05,
        #                           max_col_norm=.9,
        #                           init_bias=0.01)

        output = mlp.Softmax(layer_name='y',
                             n_classes=n_classes,
                             irange=0.1,
                             max_col_norm=1.9365)

        layers = [c16, c32, c64,
                  fc0,
                  fc1,
                  output]
        mdl = mlp.MLP(layers=layers,
                      input_space=in_space
                      )

    # layer_names = map(lambda x: x.layer_name, mdl.layers)[:-1]
    # depth = len(layer_names)
    # last_conv_layer_index = depth - 3.
    # scales = map(lambda x: min(2., (last_conv_layer_index+x)/last_conv_layer_index), range(depth))
    # probs = map(lambda x: 1/x, scales)
    # dropout_scales = dict(zip(layer_names, scales))
    # dropout_probs = dict(zip(layer_names, probs))
    # print('dropout scales: ', dropout_scales)


    trainer = sgd.SGD(learning_rate=6e-2,
                      batch_size=BATCH_SIZE,
                      learning_rule=learning_rule.Momentum(.99, nesterov_momentum=True),
                      # cost=Dropout(default_input_scale=1.,
                      #              # input_include_probs={exp_name: 0.5},
                      #              # input_scales={exp_name: 2.},
                      #              # input_scales=dropout_scales,
                      #              # input_include_probs=dropout_probs,
                      #              default_input_include_prob=1.),
                      termination_criterion=EpochCounter(max_epochs=2000),
                      monitoring_dataset={'valid': vld,
                                          'train': trn})

    velocity = learning_rule.MomentumAdjustor(final_momentum=.65,
                                              start=1,
                                              saturate=250)

    watcher = best_params.MonitorBasedSaveBest(
        channel_name='valid_y_nll',
        save_path=watcher_save_path)

    # decay = sgd.LinearDecayOverEpoch(start=1,
    #                                  saturate=350,
    #                                  decay_factor=.01)

    decay = sgd.OneOverEpoch(start=1, half_life=800)

    def create_rotator(randomize=(), center=()):
        return rotator.RotatorExtension(window=window, randomize=randomize, center=center,
                               x_offsets=range(2),
                               y_offsets=range(2),
                               median_radii=med,
                               mean_radii=mea,
                               scales=[1.01**(p*abs(p)) for p in map(lambda x: x/2., range(-11, 9))],
                               flip=True)
    rtr = create_rotator(randomize=[trn], center=[vld])
    vld_rtr = create_rotator(randomize=[vld])

    vlr = validator.Validator(vld, [vld_rtr], BATCH_SIZE, start=25, period=25,
                              # best_file_name="validator_best_model.pkl",
                              iteration_count=10)

    experiment = Train(dataset=trn,
                       model=mdl,
                       algorithm=trainer,
                       extensions=[watcher,
                                   # velocity,
                                   # rtr,
                                   # vlr,
                                   decay],
                       save_path=model_save_path,
                       save_freq=1)

    experiment.main_loop()
    logger.close()
