from __future__ import print_function
from pylearn2.models import mlp
from pylearn2.training_algorithms import sgd, learning_rule
from pylearn2.termination_criteria import EpochCounter
from pylearn2.space import Conv2DSpace
from pylearn2.train import Train
from pylearn2.train_extensions import best_params, window_flip

from mnist_dataset import MnistDataset
from dataset import Dataset, PlanktonDataset
import os
from pylearn2.utils import serial
from pylearn2.monitor import push_monitor

model_save_path = "last_model.pkl"

BATCH_SIZE = 64

# trn = MnistDataset('train')
# vld = MnistDataset('valid')
# tst = MnistDataset('test')
# nvis = 28*28

trn = PlanktonDataset(batch_size=BATCH_SIZE, which_set='train', one_hot=True)
vld = PlanktonDataset(batch_size=BATCH_SIZE, which_set='valid', one_hot=True)
vnis = 4096

n_classes = trn.n_classes
in_space = trn.in_space

watcher = best_params.MonitorBasedSaveBest(
    channel_name='valid_objective',
    save_path=model_save_path)

if os.path.isfile(model_save_path):
    print('Loading model from %s' % model_save_path)
    mdl = push_monitor(serial.load(model_save_path), 'monitor')
else:
    l1 = mlp.ConvRectifiedLinear(output_channels=128,
                                 kernel_shape=(8, 8),
                                 pool_shape=(4, 4),
                                 pool_stride=(2, 2),
                                 layer_name='l1',
                                 irange=0.05,
                                 max_kernel_norm=1.9365)
    l2 = mlp.ConvRectifiedLinear(output_channels=128,
                                 kernel_shape=(5, 5),
                                 pool_shape=(4, 4),
                                 pool_stride=(2, 2),
                                 layer_name='l2',
                                 irange=0.05,
                                 max_kernel_norm=1.9365)
    output = mlp.Softmax(layer_name='y',
                         n_classes=n_classes,
                         irange=0.5,
                         max_col_norm=1.9365)
    layers = [l1, l2, output]
    mdl = mlp.MLP(layers=layers,
                  input_space=in_space
                  )

trainer = sgd.SGD(learning_rate=5e-3,
                  batch_size=BATCH_SIZE,
                  learning_rule=learning_rule.Momentum(.5),
                  termination_criterion=EpochCounter(max_epochs=475),
                  monitoring_dataset={'valid': vld,
                                      'train': trn})


velocity = learning_rule.MomentumAdjustor(final_momentum=.65,
                                          start=1,
                                          saturate=250)

decay = sgd.LinearDecayOverEpoch(start=1,
                                 saturate=500,
                                 decay_factor=.01)

win = window_flip.WindowAndFlip(pad_randomized=8,
                                window_shape=(64, 64),
                                randomize=[trn],
                                center=[vld])

experiment = Train(dataset=trn,
                   model=mdl,
                   algorithm=trainer,
                   extensions=[watcher, decay])

experiment.main_loop()
