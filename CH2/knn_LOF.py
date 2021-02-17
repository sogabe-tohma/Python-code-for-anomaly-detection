import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import scipy.stats
from numpy import linalg as la


davis = pd.read_csv('./data/Davis_1_2.csv').values
threshold= 0.0018

    



def lof_knn():
    index_list  =   [ ]
    p_list    =   [ ]
    alpha_list   =   [ ]
    d = 0
    p  = 0
    x = davis[1:60 ,2]
    y = davis[1:60 ,3]
    dx = x - x.mean(axis = 0)
    dy = y - y.mean(axis = 0)
    data = dx.shape[0]
   
    for s in range(data):        
        distance = [] 
        for i in range(data):
            dxl = dx[i] - dx[s]
            dyl = dy[i] - dy[s]
            d2 = (dxl) ** 2 + (dyl) ** 2
            d2 = d2 ** 0.5
            distance.append(d2)
        distance_array = np.array(distance)
        index_list = sorted(range(len(distance_array)), key=lambda j: distance_array[j])

        k=20
        distance_array = np.sort(distance_array)
        for n in range(k) :
            d = d + distance_array[n]
        d = d/(k)
        for m in index_list[1:k+1]:
             p_list    =   [ ]
             print ('mis',m)
             for i in range(data):
                 xl = dx[i] - dx[m]
                 yl = dy[i] - dy[m]
                 l2 = (xl) ** 2 + (yl) ** 2
                 p_list.append(l2**0.5)
             p_list = np.array(p_list)
             p_li = np.sort(p_list)             
             for n in range(k) :
                 p = p + p_li[n]
                 print ('pis',p)        
        p = p/(k*k)
        alpha =d/p
        alpha_list.append(alpha)
    abnormals = np.array(alpha_list)
    dx = np.linspace(0, 59, 59)
    plt.scatter(np.round(dx),abnormals, marker =  ",",   c="green", alpha=0.2)
    #plt.scatter(np.round(x),abnormals, marker =  "o",   c="blue",alpha=0.8)
    plt.ylim(0.1,5)
    print ( abnormals)
    return abnormals

def abnormality_score(ap):
    result = []
    abnormality_score = ap
    data = abnormality_score.shape[0]
    index= sorted(range(len(abnormality_score)), key=lambda j: abnormality_score[j])
    for m in range(data):
        abnormality = abnormality_score[m]
        if abnormality > threshold:
            result.append(m)
    print ( result)
    print ('final results is', index)
    return result
    

ap=lof_knn()
abnormality_score(ap)
plt.show()
