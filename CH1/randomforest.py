import numpy as np
import pandas as pd
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


    def entropy(self, target):
        _, hist = np.unique(target, return_counts=True)
        ps = hist / len(target)
        return -np.sum([p * np.log2(p) for p in ps if p > 0])


    def information_gain(self, target, data_col, threshold_split):
        index_left, index_right = self.root_split(data_col, threshold_split)
        if len(index_left) == 0 or len(index_right) == 0:
            return 0

        #ジニ係数という分割基準を計算する
        original_entropy = self.entropy(target)
        e_left = self.entropy(target[index_left])
        e_right = self.entropy(target[index_right])
        n_left, n_right = len(index_left), len(index_right)
        n_total = n_left + n_right
        weighted_average_entropy = e_left * (n_left / n_total) + e_right * (n_right / n_total)
        return original_entropy - weighted_average_entropy


    def criteria_4_best(self, inputs, target):
        gain_best = -1
        threshold_split = None
        thresholds = np.unique(inputs)
        for thrs in thresholds:
            gain = self.information_gain(target, inputs, thrs)
            if gain > gain_best:
                gain_best = gain
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


class RandomForest():
    def __init__(self, t_numbers=10, split_minimum=5, depth_maximum=100, n_feats=None):
        self.t_numbers = t_numbers
        self.split_minimum = split_minimum
        self.depth_maximum = depth_maximum
        self.n_feats = n_feats
        self.use_trees = []

    def fit(self, inputs, target, sample_node_now=10):
        self.use_trees = []
        for  _ in range(self.t_numbers):
            current_tree = Tree_Decision(split_minimum=self.split_minimum,
                depth_maximum=self.depth_maximum)
            x_samp, y_samp = self.sampling_bootstrap(inputs, target, sample_node_now)

            current_tree.fit(x_samp, y_samp)
            self.use_trees.append(current_tree)

    def predict(self, inputs):
        t_predicts = np.array([current_tree.predict(inputs) for current_tree in self.use_trees])
        t_predicts = np.swapaxes(t_predicts, 0, 1)
        target_predicts = [np.mean(tree_pred) for tree_pred in t_predicts]
        return np.array(target_predicts)

    def accuracy(self,y_true, target_predicts):
        accuracy = np.sum(y_true == target_predicts) / len(y_true)
        return accuracy


    def sampling_bootstrap(self, inputs, target, sample_node_now):
        samples = inputs.shape[0]
        n_indxs = np.random.choice(samples, sample_node_now, replace=True)
        return inputs[n_indxs], target[n_indxs]


def main():
    # data and plot result
    inputs = np.array([5.0, 7.0, 12.0, 20.0, 23.0, 25.0,
                       28.0, 29.0, 34.0, 35.0, 40.0])
    target = np.array([62.0, 60.0, 83.0, 120.0, 158.0, 172.0,
                       167.0, 204.0, 189.0, 140.0, 166.0])

    plf = RandomForest(t_numbers=3, depth_maximum=2)
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
