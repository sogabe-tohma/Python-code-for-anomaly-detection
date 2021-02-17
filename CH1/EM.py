import numpy as np
import matplotlib.pyplot as plt


class GaussianMixture():
    def __init__(self, total_component):
        self.total_component = total_component


    def G_M_calcultaion(self, inputs):
        accuracy = np.linalg.inv(self.current_covs.T).T
        changed = inputs[:, :, None] - self.current_mean
        exp_calculation = np.sum(np.einsum('nik,ijk->njk', changed,
                                     accuracy) * changed, axis=1)
        return np.exp(-0.5 * exp_calculation) / np.sqrt(np.linalg.det(self.current_covs.T).T * (2 * np.pi) ** self.dimention_size)


    def fit(self, inputs, maximum_iteration=10):
        self.dimention_size = inputs.shape[1]
        self.current_weight = np.ones(self.total_component) / self.total_component
        self.current_mean = np.random.uniform(
            inputs.min(), inputs.max(), (self.dimention_size, self.total_component))
        self.current_covs = np.repeat(
            10 * np.eye(self.dimention_size), self.total_component).reshape(self.dimention_size, self.dimention_size, self.total_component)

        for i in range(maximum_iteration):
            current_params = np.hstack(
                (self.current_weight.ravel(), self.current_mean.ravel(), self.current_covs.ravel()))
            current_resps = self.belife(inputs)
            self.make_maximum(inputs, current_resps)
            if np.allclose(current_params, np.hstack((self.current_weight.ravel(), self.current_mean.ravel(), self.current_covs.ravel()))):
                break
        else:
            print("parameters have not converged")

    # E-step
    def belife(self, inputs):
        current_resps = self.current_weight * self.G_M_calcultaion(inputs)
        current_resps /= current_resps.sum(axis=-1, keepdims=True)
        return current_resps

    # M-step
    def make_maximum(self, inputs, current_resps):
        Nk = np.sum(current_resps, axis=0)
        self.current_weight = Nk / len(inputs)
        self.current_mean = np.dot(inputs.T, current_resps) / Nk
        changed = inputs[:, :, None] - self.current_mean
        self.current_covs = np.einsum('nik,njk->ijk', changed,
                              changed * np.expand_dims(current_resps, 1)) / Nk

    # 式(85)の計算
    def type_sort(self, inputs):
        joint_prob = self.current_weight * self.G_M_calcultaion(inputs)
        return np.argmax(joint_prob, axis=1)


    def probability_prediction(self, inputs):
        G_M_calcultaion = self.current_weight * self.G_M_calcultaion(inputs)
        return np.sum(G_M_calcultaion, axis=-1)


def main():
    x1 = np.random.normal(size=(100, 2)) + np.array([-5, 5])
    x2 = np.random.normal(size=(100, 2)) + np.array([5, -5])
    x3 = np.random.normal(size=(100, 2))
    inputs = np.vstack((x1, x2, x3))
    GM_model = GaussianMixture(3)
    GM_model.fit(inputs, maximum_iteration=30)
    target = GM_model.type_sort(inputs)

    data_test, target_test = np.meshgrid(
        np.linspace(-10, 10, 100), np.linspace(-10, 10, 100))
    Data_test = np.array([data_test, target_test]).reshape(2, -1).transpose()
    probs = GM_model.probability_prediction(Data_test)
    Probs = probs.reshape(100, 100)
    colors = ["red", "blue", "green"]
    plt.scatter(inputs[:, 0], inputs[:, 1], c=[colors[int(label)] for label in target])
    plt.contour(data_test, target_test, Probs)
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.show()


if __name__ == '__main__':
    main()
