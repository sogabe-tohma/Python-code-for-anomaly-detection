# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import statsmodels.graphics.api as smg
from statsmodels.graphics.tsaplots import plot_pacf
import pandas as pd
import random
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa import stattools as st
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA


#np.random.seed(1273465)

def load_data():
    df_read =pd.read_csv('discord.csv')
    return df_read.iloc[1:6000]

df_nile = load_data()
df_nile_1 = load_data()


df_nile['lag1'] = df_nile['volume'].shift(1)
df_nile['lag2'] = df_nile['volume'].shift(2)
df_nile['lag3'] = df_nile['volume'].shift(3)

df_nile = df_nile.dropna()

X_train = df_nile[['lag1', 'lag2', 'lag3']][:2000].values
X_test = df_nile[['lag1', 'lag2', 'lag3']][2000:].values

y_train = df_nile['volume'][:2000].values
y_test = df_nile['volume'][2000:].values

from sklearn import svm
r_forest = svm.SVR()


#r_forest return to dataframe
def dat_df(data,start,end,name="predict"):
    datas=[i for i in data]
    index = [i for i in range(start, end+1)]
    return pd.DataFrame(datas, index=index)

r_forest.fit(X_train, y_train)
y_train_pred = np.array(r_forest.predict(X_train))
y_train=dat_df(y_train_pred,1,2000)
y_test_pred = np.array(r_forest.predict(X_test))
y_test=dat_df(y_test_pred,2000,5995)
t_pred=pd.concat([y_train,y_test]).shift(6)

#plot the total data and return from r_forcast
#ax.plot(df_nile,figsize=(12,5), grid=False,color=['k'],marker ='o', markersize=2,markerfacecolor='w')
ax = load_data().plot(figsize=(12,2), grid=False,color=['k'],marker ='o', markersize=1,markerfacecolor='w')
#ax.plot(t_pred.values,linestyle='dotted',color='b')
ax.plot(t_pred.values,color='b')
ax.plot(t_pred.values-df_nile+2.5,linestyle='dotted',color='r',alpha=0.4)
ax.set_xlabel('mili second')
ax.set_ylabel('ms^2/Hz')
ax.get_legend().remove()

abnorm =t_pred.values-df_nile_1[3:6000]+2.5
abnorm = abnorm.dropna()
abnorm=np.array(abnorm)
abnorm.sort(axis=0)
th=abnorm[::-1][20]
np.set_printoptions(threshold=np.inf)
plt.plot([0,6000], [th,th] , color='red', linestyle='-', linewidth=0.5)
plt.show()
