import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import scipy.stats
from numpy import linalg as la


davis = pd.read_csv('./data/Davis_1_2.csv').values


class KNN2d:

    def knn2d(self):
        cx = davis[1:60 ,2]
        cy = davis[1:60 ,3]
        x = cx - cx.mean(axis = 0)
        y = cy - cy.mean(axis = 0)

        #plt.scatter(x,y)


        num = x.shape[0]
        index_list  =   [ ]
        p_list    =   [ ]
        alpha_list   =   [ ]

        for j in range(num):
            l_list = [] #k番目のデータに対するその他のデータの距離
            d = 0
            p  = 0

            for i in range(num):
                xl = x[i] - x[j]
                yl = y[i] - y[j]
                l2 = (xl) ** 2 + (yl) ** 2
                l = l2 ** 0.5

                l_list.append(l)


            l_li = np.array(l_list)
            index_list = sorted(range(len(l_li)), key=lambda k: l_li[k])
            print ('index is',index_list)

            k=20

            l_li = np.sort(l_li)

            for n in range(k) :
                d = d + l_li[n]

            d = d/(k)

            for m in index_list[1:k+1]:

                 p_list    =   [ ]
                 print ('mis',m)
                 for i in range(num):
                     xl = x[i] - x[m]
                     yl = y[i] - y[m]
                     l2 = (xl) ** 2 + (yl) ** 2

                     p_list.append(l2**0.5)

                 p_list = np.array(p_list)
                 p_li = np.sort(p_list)


                 for n in range(k) :
                     p = p + p_li[n]
                     print ('pis',p)



            p = p/(k*k)

            alpha =d/p

            print ('alpha is',  alpha)
            alpha_list.append(alpha)


        abnormals = np.array(alpha_list)
        x = np.linspace(0, 59, 59)

        plt.scatter(np.round(x),abnormals, marker =  ",",   c="green", alpha=0.2)
        #plt.scatter(np.round(x),abnormals, marker =  "o",   c="blue",alpha=0.8)
        plt.ylim(0.1,5)

        print ( abnormals)

        return abnormals


    def abnormal_decision(self, abnormals):
        treshold=0.0018
        result_list = []

        num = abnormals.shape[0]
        index= sorted(range(len(abnormals)), key=lambda k: abnormals[k])



        for i in range(num):
            abnormal = abnormals[i]
            if abnormal > treshold:
                result_list.append(i)

        print ( result_list)
        print ('final results is', index)
        return result_list

        

ss=KNN2d()
ss.knn2d()
ss.abnormal_decision(ss.knn2d())
plt.show()
