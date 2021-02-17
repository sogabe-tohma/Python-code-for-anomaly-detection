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


class AdaBoost():
    def __init__(self, t_numbers=20, depth_maximum=5):
        self.t_numbers = t_numbers
        self.depth_maximum = depth_maximum
        self.use_trees = None
        self.param_gamma = None


    def fit(self, inputs, target):
        current_data, current_target = inputs, target
        self.use_trees = []
        self.param_gamma = np.zeros((self.t_numbers, ))
        weights = np.ones((len(inputs), )) / len(inputs)

        for i in range(self.t_numbers):
            current_tree = Tree_Decision(depth_maximum=self.depth_maximum)
            all_idx = np.arange(inputs.shape[0])
            avg_weight = weights / weights.sum()
            current_ind = np.random.choice(all_idx, size=inputs.shape[0], replace=True, p=avg_weight)
            inputs = current_data[current_ind]
            target = current_target[current_ind]
            current_tree.fit(inputs, target)
            output = current_tree.predict(inputs)
            #平均誤差率em^barの計算 -> 式(28)
            error = np.absolute(output - target).reshape((-1, ))
            den = np.max(error)
            if den > 0:
                loss = error / den
            error_bar = np.sum(weights * loss)
            print('itre #%d -- error_bar=%f' % (i+1, error_bar))
            if i == 0 and error_bar == 0 or i == 0 and error_bar >= 0.5:
                self.use_trees.append(current_tree)  #
                self.param_gamma = self.param_gamma[:i+1]
                break
            if error_bar >= 0.5 or error_bar == 0:
                self.param_gamma = self.param_gamma[:i]
                break
            self.use_trees.append(current_tree)

            #各決定木に係数γの計算 -> 式(29)
            self.param_gamma[i] = error_bar / (1.0 - error_bar)
            #各データの重みwの計算 -> 式(30)
            weights *= [np.power(self.param_gamma[i], 1.0 - Ei) for Ei in loss]
            weights /= weights.sum()


    def predict(self, inputs):
        if self.param_gamma.sum() == 0:
            theta = np.ones((len(self.use_trees), )) / len(self.use_trees)
        else:
            theta = np.log(1.0 / self.param_gamma)
        current_predictors = [current_tree.predict(inputs).reshape((-1, )) for current_tree in self.use_trees]
        current_predictor = np.array(current_predictors).T
        if len(self.use_trees) == 1:
            return current_predictor
        else:
            current_ind = np.argsort(current_predictor, axis=1)
            cdf = theta[current_ind].cumsum(axis=1)
            cbf_last = cdf[:, -1].reshape((-1, 1))
            above = cdf >= (1 / 2) * cbf_last
            median_idx = above.argmax(axis=1)
            median_estimators = current_ind[np.arange(len(inputs)), median_idx]
            result = current_predictor[np.arange(
                len(inputs)), median_estimators]
            return result.reshape((-1, 1))


def main():
    # data and plot result
    inputs = np.array([5.0, 7.0, 12.0, 20.0, 23.0, 25.0,
                       28.0, 29.0, 34.0, 35.0, 40.0])
    target = np.array([62.0, 60.0, 83.0, 120.0, 158.0, 172.0,
                       167.0, 204.0, 189.0, 140.0, 166.0])

    plf = AdaBoost(t_numbers=10, depth_maximum=3)
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
