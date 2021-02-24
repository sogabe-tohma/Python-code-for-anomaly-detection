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
                weighted_average_entropy = e_left * (n_left / samples) + e_right * (n_right / samples)
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


class G_Boost():
    def __init__(self, t_numbers=5, depth_maximum=5, gamma=1, bagFraction=0.8):
        self.t_numbers = t_numbers
        self.depth_maximum = depth_maximum
        self.gamma = gamma
        self.bagFraction = bagFraction


    def fit(self, inputs, target):
        self.use_trees = []
        tree = DecisionTree(depth_maximum=self.depth_maximum)
        tree.fit(inputs, target)
        # 初期F0(x)の算出
        F0 = tree.predict(inputs)
        gradient = self.gamma * (target - F0)
        self.use_trees.append(tree)
        Fm = F0
        for _ in range(self.t_numbers - 1):
            if self.bagFraction < 1.0:
                baggings = int(round(inputs.shape[0] * self.bagFraction))
                idx = np.random.choice(range(inputs.shape[0]), baggings, replace=False)
                x = inputs[idx]
                y = gradient[idx]
            else:
                x = inputs
                y = gradient
            tree = DecisionTree(depth_maximum=self.depth_maximum)
            tree.fit(x, y)
            # 式(34)と式(35)の計算
            Fm += tree.predict(inputs)
            # 式(37)の計算
            gradient = self.gamma * (target - Fm)
            self.use_trees.append(tree)


    def predict(self, inputs):
        predicts = [tree.predict(inputs) for tree in self.use_trees]
        return np.sum(predicts, axis=0)

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
