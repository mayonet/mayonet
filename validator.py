import numpy as np
from pylearn2.train_extensions import TrainExtension
from pylearn2.monitor import Monitor
from theano import function
import sys
from pylearn2.utils import serial


class Validator(TrainExtension):

    def __init__(self, target, iterators, batch_size, start=0, period=10, iteration_count=12, best_file_name=None):
        self.target = target
        self.original = target.get_topological_view()
        self.iterators = iterators
        self.batch_size = batch_size
        self.start = start
        self.period = period
        self.iteration_count = iteration_count
        self.count = 0
        self.best = None
        self.best_file_name = best_file_name

    def setup(self, model, dataset, algorithm):
        monitor = Monitor.get_monitor(model)
        self.count = monitor.get_epochs_seen()

    def on_monitor(self, model, dataset, algorithm):
        if self.count >= self.start and (self.count-self.start) % self.period == 0:
            self.validate(model, dataset, algorithm)
        self.count += 1

    def validate(self, model, dataset, algorithm):
        sys.stdout.write("starting validation iterators\n")
        buff = self.target.get_topological_view()
        self.target.set_topological_view(self.original, self.target.view_converter.axes)
        for iterator in self.iterators:
            iterator.setup(model, dataset, algorithm)

        y = self.target.y
        y0 = np.zeros((self.target.X.shape[0], y.shape[1]))
        for i in range(self.iteration_count):
            sys.stdout.write("iteration %i\n" % i)
            X = model.get_input_space().make_batch_theano()
            Y = model.fprop(X)
            f = function([X], Y)
            for iterator in self.iterators:
                iterator.on_monitor(model, dataset, algorithm)
            yhat = []
            for j in xrange(self.target.X.shape[0] / self.batch_size):
                x_arg = self.target.X[j * self.batch_size:(j + 1) * self.batch_size, :]
                x_arg = self.target.get_topological_view(x_arg)
                yhat.append(f(x_arg.astype(X.dtype)))
            y0 += np.vstack(yhat)
        y0 /= self.iteration_count
        res = np.average(np.sum(-y*np.log(y0), axis=1))
        sys.stdout.write('validation_result: %f\n' % res)
        self.target.set_topological_view(buff, self.target.view_converter.axes)
        if self.best_file_name is not None and (self.best is None or self.best > res):
            self.best = res
            serial.save(self.best_file_name, model)
            sys.stdout.write('saved best to: %s\n' % self.best_file_name)