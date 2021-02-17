import pandas as pd
import numpy as np
import matplotlib.pyplot as plt




df = pd.read_csv("./discord.txt", sep="\t", header=None)

# 前から3001件目〜6000件のデータを対象とする
data = df.loc[0:6000, 2].reset_index(drop=True)

w = 50
l = 25
d = 10
m = 2
T = len(data)
abnorm = [0 for i in range(0, T)]




def embed(lst, dim):
    emb = np.empty((0,dim), float)
    for i in range(lst.size - dim + 1):
        tmp = np.array(lst[i:i+dim])[::-1].reshape((1,-1))
        emb = np.append( emb, tmp, axis=0)
    return emb

for s in range(l+w-1, T-d):
    H1 = embed(data[s-w-l+1:s].values, w).T
    Htest = embed(data[s-w-l+1+d:s+d].values, w).T
    U1 =  np.linalg.svd(H1)[0]
    U2 =  np.linalg.svd(Htest)[0]
    e  =  np.linalg.svd(np.dot(U1[:, 0:m].T , U2[:, 0:m]))[1]
    ab = e[0]
    print ('e', e)
    abnorm[s] = (1 - ab*ab)*2000

fig, ax1 = plt.subplots()
ax1.plot(data, color='b',alpha=0.8)
ax1.plot(abnorm, color='r',linestyle='dotted')
ax1.set_xlabel('mili second')
ax1.set_ylabel('ms^2/Hz')

abnorm_sort = np.sort(abnorm)
th=abnorm_sort[::-1][5]
plt.plot([0,6000], [th,th] , color='red', linestyle='-', linewidth=0.5)
plt.show()
