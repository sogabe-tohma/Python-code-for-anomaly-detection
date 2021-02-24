import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class MCPFilter():
    def __init__(self, particles, sigma, alpha, seed=20):
        self.particles = particles
        self.sigma = sigma
        self.alpha = alpha
        self.seed = seed


    def k_val(self, w_cumsum, idx, u):
        if not (w_cumsum < u).any():
            return 0
        else:
            return (idx[w_cumsum < u]).max() + 1


    def mc_sampling(self, weights):
        idx = np.arange(self.particles)
        u0 = np.random.uniform(0, 1/self.particles)
        U = np.arange(self.particles) / self.particles + u0
        w_cumsum = np.cumsum(weights)
        k_list = np.array([self.k_val(w_cumsum, idx, val) for val in U])
        return k_list


    def filtering(self, y):
        np.random.seed(self.seed)
        T = len(y)
        x = np.zeros((T+1, self.particles))
        x_resample = np.zeros((T+1, self.particles))
        x_init = np.random.normal(0, 1, size=self.particles)
        x_resample[0] = x_init
        x[0] = x_init
        w = np.zeros((T, self.particles))
        normalized_weight = np.zeros((T, self.particles))
        l = np.zeros(T)

        for t in range(T):
            for i in range(self.particles):
                v = np.random.normal(0, np.sqrt(self.alpha*self.sigma))
                x[t+1, i] = x_resample[t, i] + v
                w[t, i] = np.exp(-(y[t]-x[t+1, i])**2/(2*self.sigma)) / np.sqrt(2*np.pi*self.sigma)

            # 式(86)に関する計算
            normalized_weight[t] = w[t]/np.sum(w[t])
            k = self.mc_sampling(normalized_weight[t])
            x_resample[t+1] = x[t+1, k]
            l[t] = np.log(np.sum(w[t]))

        log_likelihood = np.sum(l) - T*np.log(self.particles)
        filtered_value = np.diag(np.dot(normalized_weight, x[1:].T))
        return x, filtered_value, log_likelihood


def main():
    content = pd.read_csv('./data/AirPassengers.csv')
    df = content['#Passengers']

    particles = 200
    sigma = 15
    alpha = 5

    model = MCPFilter(particles, sigma, alpha)
    X, filtered_value, log_likelihood = model.filtering(df)

    plt.figure(figsize=(12, 6))
    plt.plot(df.values, '.-k', label='data')
    plt.plot(filtered_value, '.--g', label='filtered')
    for t in range(len(df)):
        plt.scatter(np.ones(particles)*t, X[t], color='r', s=1, alpha=0.6)
    plt.title(
        f'sigma^2={sigma}, alpha^2={alpha}, log likelihood={log_likelihood :.3f}')
    plt.legend()
    plt.xlabel('input data length')
    plt.ylabel('input and generated data value')
    plt.show()


if __name__ == '__main__':
    main()
