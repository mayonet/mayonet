#!/usr/bin/env python
from pylearn2.models import mlp, maxout
from pylearn2.costs.mlp.dropout import Dropout
from pylearn2.training_algorithms import sgd, learning_rule
from pylearn2.termination_criteria import EpochCounter
from pylearn2.datasets import cifar10
from pylearn2.datasets.preprocessing import Pipeline, ZCA
from pylearn2.datasets.preprocessing import GlobalContrastNormalization
from pylearn2.space import Conv2DSpace
from pylearn2.train import Train
from pylearn2.train_extensions import best_params, window_flip
from pylearn2.utils import serial

from dataset import Dataset

BATCH_SIZE = 64

trn = Dataset(which_set='train', one_hot=True, batch_size=BATCH_SIZE)

tst = Dataset(which_set='valid', one_hot=True, batch_size=BATCH_SIZE)

in_space = Conv2DSpace(shape=(64, 64),
                       num_channels=1,
                       axes=('c', 0, 1, 'b'))

l1 = maxout.MaxoutConvC01B(layer_name='l1',
                           pad=4,
                           tied_b=1,
                           W_lr_scale=.05,
                           b_lr_scale=.05,
                           num_channels=32,
                           num_pieces=2,
                           kernel_shape=(8, 8),
                           pool_shape=(4, 4),
                           pool_stride=(2, 2),
                           irange=.005,
                           max_kernel_norm=.9,
                           # partial_sum=33
)

l2 = maxout.MaxoutConvC01B(layer_name='l2',
                           pad=3,
                           tied_b=1,
                           W_lr_scale=.05,
                           b_lr_scale=.05,
                           num_channels=64,
                           num_pieces=2,
                           kernel_shape=(8, 8),
                           pool_shape=(4, 4),
                           pool_stride=(2, 2),
                           irange=.005,
                           max_kernel_norm=1.9365,
                           # partial_sum=15
)

l3 = maxout.MaxoutConvC01B(layer_name='l3',
                           pad=3,
                           tied_b=1,
                           W_lr_scale=.05,
                           b_lr_scale=.05,
                           num_channels=64,
                           num_pieces=2,
                           kernel_shape=(5, 5),
                           pool_shape=(2, 2),
                           pool_stride=(2, 2),
                           irange=.005,
                           max_kernel_norm=1.9365)

l4 = maxout.Maxout(layer_name='l4',
                   irange=.005,
                   num_units=500,
                   num_pieces=5,
                   max_col_norm=1.9)

output = mlp.Softmax(layer_name='y',
                     n_classes=121,
                     irange=.005,
                     max_col_norm=1.9365)

layers = [l1,
          # l2,
          # l3,
          # l4,
          output]

mdl = mlp.MLP(layers,
              input_space=in_space)

trainer = sgd.SGD(learning_rate=.17,
                  batch_size=BATCH_SIZE,
                  learning_rule=learning_rule.Momentum(.5),
                  # Remember, default dropout is .5
                  cost=Dropout(input_include_probs={'l1': .8},
                               input_scales={'l1': 1.}),
                  termination_criterion=EpochCounter(max_epochs=475),
                  monitoring_dataset={'valid': tst,
                                      'train': trn})

preprocessor = Pipeline([GlobalContrastNormalization(scale=55.), ZCA()])
trn.apply_preprocessor(preprocessor=preprocessor, can_fit=True)
tst.apply_preprocessor(preprocessor=preprocessor, can_fit=False)
serial.save('/plankton/fixed64_preprocessor.pkl', preprocessor)

watcher = best_params.MonitorBasedSaveBest(
    channel_name='valid_y_misclass',
    save_path='conv_maxout1_zca.pkl')

velocity = learning_rule.MomentumAdjustor(final_momentum=.65,
                                          start=1,
                                          saturate=250)

decay = sgd.LinearDecayOverEpoch(start=1,
                                 saturate=500,
                                 decay_factor=.01)

win = window_flip.WindowAndFlip(pad_randomized=8,
                                window_shape=(64, 64),
                                randomize=[trn],
                                center=[tst])

experiment = Train(dataset=trn,
                   model=mdl,
                   algorithm=trainer,
                   extensions=[watcher, velocity, decay
                               # , win
                               ])

experiment.main_loop()
