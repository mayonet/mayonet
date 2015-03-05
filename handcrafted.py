import numpy as np
import pandas as pd
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC
from ann import *
from dataset import read_labels
from np_dataset import one_hot

floatX = 'float32'

FN = '/home/yoptar/git/subway-plankton/train_img_props.csv'

d = pd.read_csv(FN, sep='\t')

unique_labels = read_labels()
n_classes = len(unique_labels)
label_to_int = {unique_labels[i]: i for i in range(n_classes)}
y_labels = d['class'].apply(lambda row: label_to_int[row]).values
y = one_hot(y_labels)

x = np.cast[floatX](d.drop(['class', 'file_name'], axis=1).values)

for train_i, valid_i in StratifiedKFold(y_labels):
    train_x = x[train_i]
    means = train_x.mean(axis=0)
    stds = np.std(train_x, axis=0)
    train_x = (train_x - means) / stds
    valid_x = x[valid_i]
    valid_x = (valid_x - means) / stds
    train_y = y[train_i]
    valid_y = y[valid_i]
    break

# model = LinearSVC(verbose=1)
# model.fit(train_x, train_y)
# pred = model._predict_proba_lr(valid_x)
# nll = log_loss(valid_y, pred)


len_in = train_x.shape[1]
len_out = train_y.shape[1]

train_x = (train_x, train_x)
valid_x = (valid_x, valid_x)

prelu_alpha = 0.25

mlp = MLP([
    Parallel([(MLP([
                   DenseLayer(300, leaky_relu_alpha=1),
                   PReLU(prelu_alpha),
                   ])),
              (MLP([
                   DenseLayer(1500, leaky_relu_alpha=1),
                   NonLinearity(),

                   Dropout(0.5),
                   DenseLayer(1500),
                   NonLinearity(),

                   Dropout(0.5),
                   DenseLayer(1500),
                   NonLinearity(),
                   ]))]),
    Dropout(0.5),
    DenseLayer(121, leaky_relu_alpha=prelu_alpha),
    NonLinearity(activation=T.nnet.softmax)
], input_shape=(train_x[0].shape[1:], train_x[1].shape[1:]))

# mlp = MLP([
#     DenseLayer(1500, leaky_relu_alpha=1),
#     NonLinearity(),
#
#     Dropout(0.5),
#     DenseLayer(1500),
#     NonLinearity(),
#
#     Dropout(0.5),
#     DenseLayer(1500),
#     NonLinearity(),
#
#     Dropout(0.5),
#     DenseLayer(121, leaky_relu_alpha=prelu_alpha),
#     NonLinearity(activation=T.nnet.softmax)
# ], train_x.shape[1:])


## TODO move to mlp.get_updates
l2 = 0.0001  # 1e-5
learning_rate = 1e-2
momentum = 0.99
epoch_count = 500
batch_size = 100
minibatch_count = train_x[0].shape[0] // batch_size
learning_decay = 0.5 ** (1./(800 * minibatch_count))
momentum_decay = 1  # 0.5 ** (1./(300 * minibatch_count))
lr_min = 1e-6
mm_min = 0.4

method = 'adadelta+nesterov'

print('batch=%d, l2=%f, method=%s\nlr=%f, lr_decay=%f,\nmomentum=%f, momentum_decay=%f' %
      (batch_size, l2, method, learning_rate, learning_decay, momentum, momentum_decay))


tr = Trainer(mlp, batch_size, learning_rate, train_x, train_y, valid_X=valid_x, valid_y=valid_y, method=method,
             momentum=momentum, lr_decay=learning_decay, lr_min=lr_min, l2=l2, mm_decay=momentum_decay, mm_min=mm_min)

train_start = time.time()
for res in tr():
    print('{epoch}\t{train_nll:.5f}\t{test_nll:.5f}\t{epoch_time:.1f}s\t{valid_misclass:.2f}%%\t{lr}\t'
                 '{momentum}\t{l2_error}'.format(**res))

total_spent_time = time.time() - train_start
print('Trained %d epochs in %.1f seconds (%.2f seconds in average)' % (epoch_count, total_spent_time,
                                                                       total_spent_time / epoch_count))




