import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris


class SOM():
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    # 式(82)に関する計算
    def best_matching_unit(self, s):
        x_bmu = np.array([0, 0])
        minimum_distance = np.iinfo(np.int).max
        for k in range(self.structure.shape[0]):
            for j in range(self.structure.shape[1]):
                weight = self.structure[k, j, :].reshape(self.m, 1)
                distance = np.sum((weight - s) ** 2)
                if (distance < minimum_distance):
                    minimum_distance = distance
                    x_bmu = np.array([k, j])
        best_mu = self.structure[x_bmu[0], x_bmu[1], :].reshape(self.m, 1)
        return(best_mu, x_bmu, minimum_distance)


    def change_radius(self, initial_radius, i, t_cons):
        return initial_radius * np.exp(-i / t_cons)


    def lr_decay(self, learning_rate, i, iters):
        return learning_rate * np.exp(-i / iters)


    def changes(self, distance, r_data):
        return np.exp(-distance / (2 * (r_data**2)))


    def som_training(self, list_data, classes, iterations):
        self.n, self.m = list_data.shape
        network_dim = np.array([classes*2, classes*2])
        self.structure = np.random.random(((network_dim[0], network_dim[1], self.m)))
        r_init = max(network_dim[0], network_dim[1]) / 2
        t_cons = self.n / np.log(r_init)

        list_bmu = []
        list_radius = []
        list_lr = []
        list_distance = []

        for i in range(iterations):
            s = list_data[i, :].reshape(np.array([self.m, 1]))
            _, x_bmu, dist = self.best_matching_unit(s)
            list_bmu.append(x_bmu)
            list_distance.append(dist)
            r = self.change_radius(r_init, i, t_cons)
            l = self.lr_decay(self.learning_rate, i, iterations)

            list_radius.append(r)
            list_lr.append(l)

            for x in range(self.structure.shape[0]):
                for y in range(self.structure.shape[1]):
                    weight = self.structure[x, y, :].reshape(self.m, 1)
                    weight_dist = np.sum((np.array([x, y]) - x_bmu) ** 2)

                    if weight_dist <= r**2:
                        influence = self.changes(weight_dist, r)
                        weight_new = weight + (l * influence * (s - weight))
                        self.structure[x, y, :] = weight_new.reshape(1, self.m)

        list_bmu = np.array(list_bmu)
        return list_bmu, list_radius, list_lr, list_distance


    def som_builder(self, target, list_bmu, classes):
        data_2_x = np.random.randint(0, 6, self.n)
        data_2_y = np.random.randint(0, 6, self.n)

        target_color = np.zeros((self.n, 3))
        for i, v in enumerate(target):
            if (v == 0):
                target_color[i, 0] = 1
            elif (v == 1):
                target_color[i, 1] = 1
            elif (v == 2):
                target_color[i, 2] = 1

        x_noise_min = y_noise_min = -0.4
        x_noise_max = y_noise_max = 0.4
        x_noise = (x_noise_max - x_noise_min) * np.random.rand(self.n,) + x_noise_min
        y_noise = (y_noise_max - y_noise_min) * np.random.rand(self.n,) + y_noise_min
        plt_x_noise = list_bmu[:, 0] + x_noise
        plt_y_noise = list_bmu[:, 1] + y_noise
        x_noise = data_2_x + x_noise
        y_noise = data_2_y + y_noise

        e_legend = [plt.scatter(0, 0, c='r', label='setosa'),
                    plt.scatter(0, 0, c='g', label='versicolor'),
                    plt.scatter(0, 0, c='b', label='virginica')]

        plt.scatter(data_2_x, data_2_y, s=20, c=target_color)
        plt.title(f'{self.n} Inputs unsorted without noise')
        plt.legend(handles=e_legend, loc=1)
        plt.show()

        plt.scatter(x_noise, y_noise, s=20, c=target_color)
        plt.title(f'{self.n} Inputs unsorted with noise')
        plt.legend(handles=e_legend, loc=1)
        plt.show()

        plt.scatter(list_bmu[:, 0], list_bmu[:, 1], s=20, c=target_color)
        plt.title(f'{self.n} Inputs sorted without noise')
        plt.legend(handles=e_legend, loc=1)
        plt.show()

        plt.scatter(plt_x_noise, plt_y_noise, s=20, c=target_color)
        plt.title(f'{self.n} Inputs sorted with noise')
        plt.legend(handles=e_legend, loc=1)
        plt.show()


def main():
    data = load_iris()
    inputs = data['data']
    inputs = inputs/inputs.max()
    target = data['target']
    classes = 3
    learning_rate = 0.3

    som = SOM(learning_rate)
    best_mu, r_data, rate, squqred_dist = som.som_training(inputs, classes, 150)
    som.som_builder(target, best_mu, classes)

    plt.title('radius_evolution')
    plt.xlabel('iterations')
    plt.ylabel('radius_size')
    plt.plot(r_data)
    plt.show()

    plt.title('learning_rate_evolution')
    plt.xlabel('iterations')
    plt.ylabel('learning_rate')
    plt.plot(rate)
    plt.show()

    plt.title('best_matching_unit_3D_distance')
    plt.xlabel('iterations')
    plt.ylabel('smallest_distance_squared')
    plt.plot(squqred_dist)
    plt.show()


if __name__ == '__main__':
    main()
