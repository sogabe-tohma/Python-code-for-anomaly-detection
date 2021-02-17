import numpy as np
import matplotlib.pyplot as plt


class Tsne():
    def __init__(self, data, label, TSNE=True, learning_rate=0.01, MOMENTUM=0.9, iter=100, seed=123):
        self.TSNE = TSNE
        self.iteration = iter
        self.label = label
        self.X = data
        self.learning_rate = learning_rate
        self.momentum = MOMENTUM
        self.seed = seed


    def connect_p(self, X, p_information=2):
        input_sum = np.sum(np.square(X), 1)
        neg_distance = np.add(np.add(-2 * np.dot(X, X.T), input_sum).T, input_sum)
        distance = -neg_distance
        total = self.sigma_opt(distance, p_information)
        _P = self.matrix_4_p(distance, total)
        P = (_P + _P.T) / (2. * _P.shape[0])
        return P

    # 式(76)の計算
    def matrix_4_p(self, distance, total=None, idx_zero=None):
        if total is not None:
            q_sigma = 2. * np.square(total.reshape((-1, 1)))
            return self.cal_softmax(distance / q_sigma, idx_zero=idx_zero)
        else:
            return self.cal_softmax(distance, idx_zero=idx_zero)

    # 式(78)のデータ間の距離の計算
    def connect_q(self, y):
        input_sum = np.sum(np.square(y), 1)
        neg_distance = np.add(np.add(-2 * np.dot(y, y.T), input_sum).T, input_sum)
        distance = -neg_distance
        power_distance = np.exp(distance)
        np.fill_diagonal(power_distance, 0.)
        return power_distance / np.sum(power_distance), None

    # 式(78)の計算：t-分布
    def tene_4_q(self, y):
        input_sum = np.sum(np.square(y), 1)
        neg_distance = np.add(np.add(-2 * np.dot(y, y.T), input_sum).T, input_sum)
        distance = -neg_distance
        inverse_distance = np.power(1. - distance, -1)
        np.fill_diagonal(inverse_distance, 0.)
        return inverse_distance / np.sum(inverse_distance), inverse_distance


    def cal_softmax(self, X, d_zero=True, idx_zero=None):
        exp_x = np.exp(X - np.max(X, axis=1).reshape([-1, 1]))
        if idx_zero is None:
            if d_zero:
                np.fill_diagonal(exp_x, 0.)
        else:
            exp_x[:, idx_zero] = 0.
        exp_x = exp_x + 1e-8
        return exp_x / exp_x.sum(axis=1).reshape([-1, 1])


    def _grad_now(self, Q, Y, distance):
        y_diffs = np.expand_dims(Y, 1) - np.expand_dims(Y, 0)
        dist_expanded = np.expand_dims(distance, 2)
        y_diffs_wt = y_diffs * dist_expanded
        pq_diff = self.connect_p(self.X) - Q
        pq_expanded = np.expand_dims(pq_diff, 2)
        grad_now = 4. * (pq_expanded * y_diffs_wt).sum(1)
        return grad_now


    def _infos(self, p_matrix):
        p_information = 2 ** -np.sum(p_matrix * np.log2(p_matrix), 1)
        return p_information


    def _information(self, distance, total, idx_zero):
        return self._infos(self.matrix_4_p(distance, total, idx_zero))


    def sigma_opt(self, distance, target):
        total = []
        for i in range(distance.shape[0]):
            def eval_fn(sigma): return self._information(
                distance[i:i+1, :], np.array(sigma), i)
            correct_sigma = self._search(eval_fn, target)
            total.append(correct_sigma)
        return np.array(total)


    def tsne(self):
        self.grad_now_fn = self._grad_now if self.TSNE else self.grad_now_sym
        self.q_fn = self.tene_4_q if self.TSNE else self.connect_q
        rng = np.random.RandomState(self.seed)
        Y = rng.normal(0., 0.0001, [self.X.shape[0], 2])
        if self.momentum:
            Y_m2 = Y.copy()
            Y_m1 = Y.copy()
        for _ in range(self.iteration):
            Q, distance = self.q_fn(Y)
            grad_nows = self.grad_now_fn(Q, Y, distance)
            Y = Y - self.learning_rate * grad_nows
            if self.momentum:
                Y += self.momentum * (Y_m1 - Y_m2)
                Y_m2 = Y_m1.copy()
                Y_m1 = Y.copy()
        return Y


    def _search(self, eval_fn, target, tol=1e-10, max_iter=10000,
                lower=1e-20, upper=1000.):
        for _ in range(max_iter):
            guess = (lower + upper) / 2.
            val = eval_fn(guess)
            if val > target:
                upper = guess
            else:
                lower = guess
            if np.abs(val - target) <= tol:
                break
            return guess


    def grad_now_sym(self, P, Q, Y):
        y_diffs = np.expand_dims(Y, 1) - np.expand_dims(Y, 0)
        pq_diff = P - Q
        pq_expanded = np.expand_dims(pq_diff, 2)
        grad_now = 4. * (pq_expanded * y_diffs).sum(1)
        return grad_now


    def plt_tsne(self, plt_data, title='', ms=6, ax=None, alpha=1.0,
                 legend=True):
        target = list(np.unique(self.label))
        write = 'os' * len(target)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(target)))
        for i, cls in enumerate(target):
            mark = write[i]
            ax.plot(np.zeros(1), plt_data[self.label == cls], marker=mark,
                    linestyle='', ms=ms, label=str(cls), alpha=alpha, color=colors[i],
                    markeredgecolor='black', markeredgewidth=0.4)
        if legend:
            ax.legend()
        ax.title.set_text(title)
        return ax


    def plt_input(self, plt_data, title='', ms=6, ax=None, alpha=1.0,
                  legend=True):
        target = list(np.unique(self.label))
        write = 'os' * len(self.label)
        for i, cls in enumerate(target):
            mark = write[i]
            ax.plot(plt_data[cls][0], plt_data[cls][1], marker=mark,
                    linestyle='', ms=ms, label=str(cls), alpha=alpha,
                    markeredgecolor='black', markeredgewidth=0.4)
        ax.set_xlim(-2, 12)
        ax.set_ylim(-2, 12)
        ax.title.set_text(title)
        ax.legend()
        return ax


def main():
    x = [2.1, 1.1, 8.1, 0.9, 1.5, 1.5, 9.4, 1.3]
    y = [8.1, 7.5, 2.8, 1.7, 8.5, 1.8, 2.8, 2.2]
    inputs = np.array((x, y)).T
    label = np.arange(inputs.shape[0])
    TSNE = Tsne(inputs, label, TSNE=True, seed=123)
    results = TSNE.tsne()

    _, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    TSNE.plt_tsne(results, title='Tsne result', ax=ax[1])
    TSNE.plt_input(inputs, title='Data set', ax=ax[0])
    plt.show()


if __name__ == '__main__':
    main()
