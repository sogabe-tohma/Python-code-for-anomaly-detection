import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt


class Tree_Decision():
    def __init__(self, split_minimum=2, depth_maximum=100):
        self.split_minimum = split_minimum
        self.depth_maximum = depth_maximum
        self.root_node = None


    def root_split(self, data_col, threshold_split):
        index_left = np.argwhere(data_col <= threshold_split).flatten()
        index_right = np.argwhere(data_col > threshold_split).flatten()
        return index_left, index_right


    def tree_arrangement(self, inputs, t_node):
        if t_node.is_n_leaf():
            return t_node.val
        if inputs <= t_node.thrs:
            return self.tree_arrangement(inputs, t_node.left)
        return self.tree_arrangement(inputs, t_node.right)


    def var(self, target, data_col, threshold_split):
        index_left, index_right = self.root_split(data_col, threshold_split)
        if len(index_left) == 0 or len(index_right) == 0:
            return 900
        return (np.std(target[index_left]) + np.std(target[index_right])) / 2


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
        return np.array([self.tree_arrangement(input, self.root_node) for input in inputs])


class Tree_Node():
    def __init__(self, thrs=None, left=None, right=None, *, val=None):
        self.thrs = thrs
        self.left = left
        self.right = right
        self.val = val


    def is_n_leaf(self):
        return self.val is not None


class G_Boost():
    def __init__(self, t_numbers=5, depth_maximum=5, p_gamma=1, bagging_fraction=0.8):
        self.t_numbers = t_numbers
        self.depth_maximum = depth_maximum
        self.p_gamma = p_gamma
        self.bagging_fraction = bagging_fraction
        self.use_trees = None


    def predict(self, inputs):
        outputs = [current_trees.predict(inputs)
                   for current_trees in self.use_trees]
        return np.sum(outputs, axis=0)


    def fit(self, inputs, targets):
        self.use_trees = []
        current_trees = Tree_Decision(depth_maximum=self.depth_maximum)
        current_trees.fit(inputs, targets)
        # 初期F0(x)の算出
        c_out = current_trees.predict(inputs)
        current_grad = self.p_gamma * (targets - c_out)
        self.use_trees.append(current_trees)
        c_target = c_out
        for _ in range(self.t_numbers - 1):
            current_input = inputs
            current_target = current_grad
            if self.bagging_fraction < 1.0:
                baggings = int(round(inputs.shape[0] * self.bagging_fraction))
                index = random.sample(range(inputs.shape[0]), baggings)
                current_input = current_input[index]
                current_target = current_target[index]
            current_trees = Tree_Decision(depth_maximum=self.depth_maximum)
            current_trees.fit(current_input, current_target)
            # 式(34)と式(35)の計算
            c_target += current_trees.predict(inputs)
            # 式(37)の計算
            current_grad = self.p_gamma * (targets - c_target)
            self.use_trees.append(current_trees)
            if np.all(current_grad == 0):
                break

def main():
    # data and plot result
    inputs = np.array([5.0, 7.0, 12.0, 20.0, 23.0, 25.0,
                       28.0, 29.0, 34.0, 35.0, 40.0])
    target = np.array([62.0, 60.0, 83.0, 120.0, 158.0, 172.0,
                       167.0, 204.0, 189.0, 140.0, 166.0])

    plf = G_Boost(t_numbers=5, depth_maximum=2)
    plf.fit(inputs, target)
    y_pred = plf.predict(inputs)
    print(y_pred)
    plt.scatter(inputs, target, label='data')
    plt.step(inputs, y_pred, color='orange', label='prediction')
    plt.ylim(10, 210)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
