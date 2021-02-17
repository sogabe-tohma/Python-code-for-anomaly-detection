# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame, Series
from numpy.random import rand, multivariate_normal
from sklearn import datasets

iris = datasets.load_iris()
nagasa = iris.data[:,2]
haba = iris.data[:,3]

X = iris.data[:,2:4]
N = 150

def centering(X, N):
    #中心化を行う行列
    H = np.eye(N) - 1./ N * np.ones([N, N])
    return DataFrame(np.dot(H, X) ,columns=['x','y'])

df_center = centering(X, N)
plt.scatter(df_center[0:50]["x"], df_center[0:50]["y"], c='blue', marker='o')
plt.scatter(df_center[50:100]["x"],df_center[50:100]["y"], c='red', marker='o')
plt.scatter(df_center[100:150]["x"],df_center[100:150]["y"], c='green', marker='o')
plt.show()


def PCA(X):
    #標本の散布行列
    C = np.dot(X.T, X)
    #固有値・規格化された固有ベクトル
    w, v = np.linalg.eigh(C)
    #固有値を降順に並び替える
    #sortは基本昇順
    index = np.argsort(w)[::-1]
    #固有ベクトルを並び替える
    T_pca = v[index]
    return T_pca
T_pca = PCA(df_center)

def f_pca(x):    #第一主成分の軸
    y1 = T_pca[0][1] / T_pca[0][0] * x
    y2 = T_pca[1][1] / T_pca[1][0] * x
    return y1,y2
#アスペクト比を調整
#plt.figure(figsize=(6,6))
#第一主成分
linex = np.arange(df_center["x"].min()*0.1,df_center["x"].max()*0.1,0.01)
liney1,liney2 = f_pca(linex)                                    #求めたfを使って直線をかく
plt.plot(linex, liney1, color='red')
plt.plot(linex, liney2, color='blue')
plt.scatter(df_center[0:50]["x"], df_center[0:50]["y"], c='blue', marker='o')
plt.scatter(df_center[50:100]["x"],df_center[50:100]["y"], c='red', marker='o')
plt.scatter(df_center[100:150]["x"],df_center[100:150]["y"], c='green', marker='o')
plt.show()
