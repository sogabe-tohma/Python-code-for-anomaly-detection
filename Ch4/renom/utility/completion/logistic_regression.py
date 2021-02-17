from __future__ import division, print_function
import numpy as np
import renom as rm
from sklearn.preprocessing import LabelBinarizer
from renom.optimizer import Sgd, Adam
from renom.cuda import set_cuda_active


class LogisticRegression(object):
    def __init__(self, x, y, batch=64, epoch=50, optimizer=Sgd):
        self.lb = LabelBinarizer().fit(y)
        self.batch = batch
        self.epoch = epoch
        self.optimizer = optimizer()
        self.network = rm.Sequential([
            rm.Dense(1)
        ])

    def fit(self, x, y):
        N = len(x)
        labels = self.lb.transform(y)
        for i in range(self.epoch):
            perm = np.random.permutation(N)
            for j in range(N // self.batch):
                train_batch = x[perm[j * self.batch:(j + 1) * self.batch]]
                labels_batch = labels[perm[j * self.batch:(j + 1) * self.batch]]
                with self.network.train():
                    z = self.network(train_batch)
                    loss = rm.sigmoid_cross_entropy(z, labels_batch)
                loss.grad().update(self.optimizer)

    def predict(self, x):
        output_network = list(map(int, rm.sigmoid(self.network(x)).as_ndarray() > 0.5))
        result_array = [self.lb.classes_[output] for output in output_network]
        return np.array(result_array)


class MulticlassLogisticRegression(object):
    def __init__(self, x, y, batch=64, epoch=50, optimizer=Sgd):
        self.lb = LabelBinarizer().fit(y)
        self.batch = batch
        self.epoch = epoch
        self.optimizer = optimizer()
        self.network = rm.Sequential([
            rm.Dense(len(self.lb.classes_))
        ])

    def fit(self, x, y):
        N = len(x)
        labels = self.lb.transform(y)
        for i in range(self.epoch):
            perm = np.random.permutation(N)
            for j in range(N // self.batch):
                train_batch = x[perm[j * self.batch:(j + 1) * self.batch]]
                labels_batch = labels[perm[j * self.batch:(j + 1) * self.batch]]
                with self.network.train():
                    z = self.network(train_batch)
                    loss = rm.softmax_cross_entropy(z, labels_batch)
                loss.grad().update(self.optimizer)

    def predict(self, x):
        output_network = np.argmax(rm.softmax(self.network(x)).as_ndarray(), axis=1)
        result_array = [self.lb.classes_[output] for output in output_network]
        return np.array(result_array)
