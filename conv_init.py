train = open('conv_init.yaml', 'r').read()
train_params = {'batch_size': 100}
train = train % (train_params)

from pylearn2.config import yaml_parse
train = yaml_parse.load(train)
train.main_loop()