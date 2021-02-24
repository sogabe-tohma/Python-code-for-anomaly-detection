import numpy as np
import matplotlib.pyplot as plt


class DecisionTree():
    def __init__(self, split_minimum=2, depth_maximum=100):
        self.split_minimum = split_minimum
        self.depth_maximum = depth_maximum


    def tree_arrangement(self, inputs, node):
        if node.val is not None:
            return node.val
        if inputs <= node.thrs:
            return self.tree_arrangement(inputs, node.left)
        return self.tree_arrangement(inputs, node.right)


    def entropy(self, target):
        _, hist = np.unique(target, return_counts=True)
        p = hist / len(target)
        return -np.sum(p * np.log2(p))


    def tree_growth(self, inputs, target, depth=0):
        samples = inputs.shape[0]
        if depth >= self.depth_maximum or samples < self.split_minimum:
            return Tree_Node(val=np.mean(target))

        thresholds = np.unique(inputs)
        best_gain = -1
        for th in thresholds:
            idx_left = np.where(inputs <= th)
            idx_right = np.where(inputs > th)
            if len(idx_left) == 0 or len(idx_right) == 0:
                gain = 0
            else:
                original_entropy = self.entropy(target)
                e_left = self.entropy(target[idx_left])
                e_right = self.entropy(target[idx_right])
                n_left, n_right = len(idx_left), len(idx_right)
                weighted_average_entropy = e_left * (n_left / samples)\
                                + e_right * (n_right / samples)
                gain = original_entropy - weighted_average_entropy
            if gain > best_gain:
                index_left = idx_left
                index_right = idx_right
                best_gain = gain
                threshhold_best = th

        if best_gain == 0:
            return Tree_Node(val=np.mean(target))

        left_node = self.tree_growth(inputs[index_left], target[index_left], depth+1)
        right_node = self.tree_growth(inputs[index_right], target[index_right], depth+1)
        return Tree_Node(threshhold_best, left_node, right_node)


    def fit(self, inputs, target):
        self.root_node = self.tree_growth(inputs, target)


    def predict(self, inputs):
        return np.array([self.tree_arrangement(input_, self.root_node) for input_ in inputs])


class Tree_Node():
    def __init__(self, thrs=None, left=None, right=None, *, val=None):
        self.thrs = thrs
        self.left = left
        self.right = right
        self.val = val


class AdaBoost():
    def __init__(self, t_numbers=20, depth_maximum=5):
        self.t_numbers = t_numbers
        self.depth_maximum = depth_maximum


    def fit(self, inputs, target):
        self.use_trees = []
        self.gamma = np.zeros(self.t_numbers)
        weights = np.ones(inputs.shape[0]) / inputs.shape[0]
        all_idx = np.arange(inputs.shape[0])

        for i in range(self.t_numbers):
            tree = DecisionTree(depth_maximum=self.depth_maximum)
            avg_weight = weights / weights.sum()
            idx = np.random.choice(all_idx, size=inputs.shape[0], replace=True, p=avg_weight)
            tree.fit(inputs[idx], target[idx])
            output = tree.predict(inputs[idx])
            #平均誤差率em^barの計算 -> 式(28)
            error = abs(output - target[idx])
            loss = error / (max(error) + 1e-50)
            error_bar = np.sum(weights * loss)
            print(f'tree #{i+1} : error_bar = {error_bar}')

            if error_bar == 0 or error_bar >= 0.5:
                if i == 0:
                    self.use_trees.append(tree)
                    self.gamma = self.gamma[:i+1]
                    break
                else:
                    self.gamma = self.gamma[:i]
                    break

            self.use_trees.append(tree)
            #各決定木に係数γの計算 -> 式(29)
            self.gamma[i] = error_bar / (1.0 - error_bar)
            #各データの重みwの計算 -> 式(30)
            weights *= [np.power(self.gamma[i], 1.0 - Ei) for Ei in loss]
            weights /= weights.sum()


    def predict(self, inputs):
        predicts = np.array([tree.predict(inputs) for tree in self.use_trees])
        if len(self.use_trees) == 1:
            return predicts[0]
        else:
            theta = np.log(1.0 / self.gamma)
            idx = np.argsort(predicts, axis=0)
            cdf = theta[idx].cumsum(axis=0)
            above = cdf >= theta.sum()/2
            median_idx = above.argmax(axis=0)
            median_estimators = np.diag(idx[median_idx])
            return np.diag(predicts[median_estimators])


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
