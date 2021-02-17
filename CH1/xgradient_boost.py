import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt


class Tree_Decision():
    def __init__(self, split_minimum=2, depth_maximum=100, _lambda=0.1, gamma=0.1):
        self.split_minimum = split_minimum
        self.depth_maximum = depth_maximum
        self.root_node = None
        self._lambda = _lambda
        self.gamma = gamma


    def root_split(self, data_col, threshold_split):
        index_left = np.argwhere(data_col <= threshold_split).flatten()
        index_right = np.argwhere(data_col > threshold_split).flatten()
        return index_left, index_right


    def tree_arrangement(self, inputs, t_node):
        if t_node.is_leaf_node():
            return t_node.val
        if inputs <= t_node.thrs:
            return self.tree_arrangement(inputs, t_node.left)
        return self.tree_arrangement(inputs, t_node.right)


    def var(self, target, data_col, threshold_split):
        index_left, index_right = self.root_split(data_col, threshold_split)
        if len(index_left) == 0 or len(index_right) == 0:
            return 900
        var_left = np.sqrt(np.sum((target[index_left] - target[index_left].mean())**2) / (
            target[index_left].shape[0] + self._lambda))
        var_right = np.sqrt(np.sum((target[index_right] - target[index_right].mean())**2) / (
            target[index_right].shape[0] + self._lambda))
        return (var_left + target[index_left].shape[0]*self.gamma + var_right + target[index_right].shape[0]*self.gamma) / 2


    def criteria_4_best(self, inputs, target):
        variance_best = 1000
        threshold_split = None
        thresholds = np.unique(inputs)
        for thrs in thresholds:
            variance = self.var(target, inputs, thrs)
            if variance < variance_best:
                variance_best = variance
                threshold_split = thrs
        return threshold_split


    def tree_growth(self, inputs, target, t_depth=0):
        samples = inputs.shape[0]
        if (t_depth >= self.depth_maximum or samples < self.split_minimum):
            leaf_cur_value = np.mean(target)
            return Tree_Node(val=leaf_cur_value)

        threshhold_best = self.criteria_4_best(inputs, target)
        index_left, index_right = self.root_split(inputs, threshhold_best)
        left_side = self.tree_growth(inputs[index_left], target[index_left], t_depth+1)
        right_side = self.tree_growth(inputs[index_right], target[index_right], t_depth+1)
        return Tree_Node(threshhold_best, left_side, right_side)


    def fit(self, inputs, target):
        self.root_node = self.tree_growth(inputs, target)


    def predict(self, inputs):
        return np.array([self.tree_arrangement(inputs, self.root_node) for inputs in inputs])


class Tree_Node():
    def __init__(self, thrs=None, left=None, right=None, *, val=None):
        self.thrs = thrs
        self.left = left
        self.right = right
        self.val = val


    def is_leaf_node(self):
        return self.val is not None


class XGBoost:
    def __init__(self, t_numbers=5, depth_maximum=3, alpha=1, bagFraction=0.8, _lambda=0.1, gamma=0.1):
        self.t_numbers = t_numbers
        self.depth_maximum = depth_maximum
        self.alpha = alpha
        self.bagFraction = bagFraction
        self.use_trees = None
        self._lambda = _lambda
        self.gamma = gamma


    def fit(self, inputs, target):
        self.use_trees = []
        current_tree = Tree_Decision(
            depth_maximum=self.depth_maximum, _lambda=self._lambda, gamma=self.gamma)
        current_tree.fit(inputs, target)
        # 初期F0(x)の算出
        outs = current_tree.predict(inputs)
        cur_grad = self.alpha * (target - outs)
        c_target = outs
        self.use_trees.append(current_tree)
        for _ in range(self.t_numbers - 1):
            train_x = inputs
            train_y = cur_grad
            if self.bagFraction < 1.0:
                baggings = int(round(inputs.shape[0] * self.bagFraction))
                index = random.sample(range(inputs.shape[0]), baggings)
                train_x = train_x[index]
                train_y = train_y[index]
            current_tree = Tree_Decision(
                depth_maximum=self.depth_maximum, _lambda=self._lambda, gamma=self.gamma)
            current_tree.fit(train_x, train_y)
            # 式(45)の計算
            c_target += current_tree.predict(inputs)
            # 式(43)の計算
            cur_grad = self.alpha * (target - c_target)
            self.use_trees.append(current_tree)
            if np.all(cur_grad == 0):
                break


    def predict(self, inputs):
        z = [current_tree.predict(inputs) for current_tree in self.use_trees]
        return np.sum(z, axis=0)


def main():
    # data and plot result
    inputs = np.array([5.0, 7.0, 12.0, 20.0, 23.0, 25.0,
                       28.0, 29.0, 34.0, 35.0, 40.0])
    target = np.array([62.0, 60.0, 83.0, 120.0, 158.0, 172.0,
                       167.0, 204.0, 189.0, 140.0, 166.0])

    plf = XGBoost(t_numbers=5, depth_maximum=3, _lambda=0, gamma=0)
    plf.fit(inputs, target)
    y_pred = plf.predict(inputs)
    print(y_pred)
    plt.scatter(inputs, target, label='data')
    plt.step(inputs, y_pred, color='orange', label='prediction')
    plt.ylim(10,210)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
