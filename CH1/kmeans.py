import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import copy
from sklearn.datasets import load_iris


class Kmeans():
    def __init__(self, inputs, K=3):
        self.inputs = inputs
        self.n = inputs.shape[0]
        self.K = K
        self.clusters = np.zeros(self.n)
        self.distances = np.zeros((self.n, self.K))
        self.centers = self.cal_centers()
        self.centers_old = np.zeros(self.centers.shape)
        self.centers_new = self.centers.copy()


    def cal_centers(self):
        return np.random.randn(self.K, self.inputs.shape[1]) * self.inputs.std(axis=0) + self.inputs.mean(axis=0)


    def update_centers(self):
        while self.error() != 0:
            for i in range(self.K):
                self.distances[:, i] = np.linalg.norm(
                    self.inputs - self.centers[i], axis=1)
            # 式(83)に関する計算
            self.clusters = np.argmin(self.distances, axis=1)
            self.centers_old = self.centers_new.copy()
            # 式(84)に関する計算
            for i in range(self.K):
                self.centers_new[i] = np.mean(
                    self.inputs[self.clusters == i], axis=0)
        return self.centers_new


    def error(self):
        return np.linalg.norm(self.centers_new - self.centers_old)


def main():
    data = load_iris()
    inputs = data['data']
    target = data['target']
    kmeans = Kmeans(inputs)
    centers = kmeans.update_centers()

    plt.figure(figsize=(15, 7))
    colors = ['orange', 'blue', 'green']
    for i in range(inputs.shape[0]):
        plt.scatter(inputs[i, 0], inputs[i, 1], s=50, color=colors[int(target[i])])
    plt.scatter(centers[:, 0], centers[:, 1], marker='*', c='r', s=300)
    plt.show()


if __name__ == '__main__':
    main()
