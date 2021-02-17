# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors


def main():
    data = np.loadtxt("discord.txt",delimiter="\t")

    train_data = data[100:2000, 2]
    test_data = data[2001:6000, 2]

    #train_data = moving_average(train_data, 20)

    #test_data = moving_average(test_data, 20)

    width = 5
    nk = 1

    train = embed(train_data, width)
    test = embed(test_data, width)
    neigh = NearestNeighbors(n_neighbors=nk)
    neigh.fit(train)
    d = neigh.kneighbors(test)[0]
    d = np.mean(d, axis=1)
    mx = np.max(d)
    d = d / mx


    # プロット
    test_for_plot = data[2001+width:6000, 2]

    fig = plt.figure()

    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()

    #p1, = ax1.plot(d, '-m',linewidth = 1, linestyle="dotted" )
    p1, = ax1.plot(d, '-b',linewidth = 1 )

    ax1.set_ylim(0, 4.2)
    ax1.set_xlim(0, 4000)
    p2, = ax2.plot(test_for_plot, '-k')

    ax2.set_ylim(0, 8.0)
    ax2.set_xlim(0, 4000)

    plt.show()



def moving_average(a, n) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def embed(lst, dim):
    emb = np.empty((0,dim), float)
    for i in range(lst.size - dim + 1):
        tmp = np.array(lst[i:i+dim])[::-1].reshape((1,-1))
        emb = np.append( emb, tmp, axis=0)
    return emb

if __name__ == '__main__':
    main()
