import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class MCPFilter():
    def __init__(self, particles, sigma, alpha):
        self.particles = particles
        self.sigma = sigma
        self.alpha = alpha


    def k_val(self, w_cumsum, idx, u):
        if np.any(w_cumsum < u) == False:
            return 0
        k = np.max(idx[w_cumsum < u])
        return k+1


    def sampling(self, weights):
        idx = np.asanyarray(range(self.particles))
        u0 = np.random.uniform(0, 1/self.particles)
        U = [1/self.particles*i + u0 for i in range(self.particles)]
        w_cumsum = np.cumsum(weights)
        k_list = np.asanyarray([self.k_val(w_cumsum, idx, val) for val in U])
        for i, u in enumerate(U):
            k = self.k_val(w_cumsum, idx, u)
            k_list[i] = k
        return k_list


    def p_filter(self, y, seed=20):
        log_likelihood = -np.inf
        np.random.seed(seed)
        T = len(y)
        x = np.zeros((T+1, self.particles))
        x_resampled = np.zeros((T+1, self.particles))
        initial_x = np.random.normal(0, 1, size=self.particles)
        x_resampled[0] = initial_x
        x[0] = initial_x
        w = np.zeros((T, self.particles))
        w_normed = np.zeros((T, self.particles))
        l = np.zeros(T)

        for t in range(T):
            for i in range(self.particles):
                v = np.random.normal(0, np.sqrt(self.alpha*self.sigma))
                x[t+1, i] = x_resampled[t, i] + v
                w[t, i] = (np.sqrt(2*np.pi*self.sigma))**(-1) * \
                    np.exp(-(y[t]-x[t+1, i])**2/(2*self.sigma))

            # 式(86)に関する計算
            w_normed[t] = w[t]/np.sum(w[t])
            l[t] = np.log(np.sum(w[t]))
            k = self.sampling(w_normed[t])
            x_resampled[t+1] = x[t+1, k]

        log_likelihood = np.sum(l) - T*np.log(self.particles)
        filtered_value = np.diag(np.dot(w_normed, x[1:].T))
        return x, filtered_value, log_likelihood


def main():
    content = pd.read_csv('./data/AirPassengers.csv')
    df = content['#Passengers']

    particles = 200
    sigma = 15
    alpha = 5

    model = MCPFilter(particles, sigma, alpha)
    X, filtered_value, log_likelihood = model.p_filter(df)

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
