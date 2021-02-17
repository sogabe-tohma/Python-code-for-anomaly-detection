# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import scipy.stats
from numpy import linalg as la
from scipy.stats import norm
import seaborn as sns



davis = pd.read_csv('./data/Davis.csv').values
x = davis[: ,2:3]

#plt.hist( x,color='blue',bins=200)

sns.distplot(x, fit = norm, color='k', kde=False, bins =50,rug=True)



# 平均ベクトル
#mx = x.mean(axis = 0)

# 中心化データ
#xc = x - mx

# 標本分散ベクトル
#sx = ( xc.T.dot(xc) / x[:,0].size ).astype(float)
#sx = x.std()

# 標本分散ベクトルの逆数
#sx_inv= np.linalg.inv(sx)



#plt.hist( x,color='blue',bins=200)

plt.show()



"""
# 異常度
ap = np.dot(xc, np.linalg.inv(sx)) * xc

#plt.hist( ap,color='blue',bins=200)


# 閾値:分位点法
th = 4.27

# 閾値
#th = sp.stats.chi2.ppf(0.98,1)


plt.scatter(np.arange(ap.size), ap , color='b')

plt.plot([0,200], [th,th] , color='red', linestyle='-', linewidth=1)
plt.ylim(0,55)
plt.show()
"""