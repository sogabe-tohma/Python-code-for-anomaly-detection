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

df_discord = load_data()

df_discord_1 = load_data()

df_discord['lag1'] = df_discord['volume'].shift(1)
df_discord['lag2'] = df_discord['volume'].shift(2)
df_discord['lag3'] = df_discord['volume'].shift(3)

df_discord = df_discord.dropna()
X_train = df_discord[['lag1', 'lag2', 'lag3']][:2000].values
X_test = df_discord[['lag1', 'lag2', 'lag3']][2000:].values
y_train =df_discord['volume'][:2000].values
y_test = df_discord['volume'][2000:].values
#r_forest return to dataframe
def dat_df(data,start,end,name="predict"):
    datas=[i for i in data]
    index = [i for i in range(start, end+1)]
    return pd.DataFrame(datas, index=index)
from sklearn.ensemble import RandomForestRegressor
r_forest = RandomForestRegressor(
            n_estimators=20,
            criterion='mse',
            random_state=1,
            n_jobs=-1
)

r_forest.fit(X_train, y_train)
y_train_pred = np.array(r_forest.predict(X_train))
y_train=dat_df(y_train_pred,1,2000)
y_test_pred = np.array(r_forest.predict(X_test))
y_test=dat_df(y_test_pred,2000,5995)
t_pred=pd.concat([y_train,y_test]).shift(6)
np.set_printoptions(threshold=np.inf)
print (y_train)


"""
#plot the total data and return from r_forcast
#ax.plot(df_discord,figsize=(12,5), grid=False,color=['k'],marker ='o', markersize=2,markerfacecolor='w')

ax = load_data().plot(figsize=(12,2), grid=False,color=['k'],marker ='o', markersize=2,markerfacecolor='w')
ax.plot(t_pred.values,linestyle='dotted',color='b')
ax.plot(t_pred.values-df_discord_1[3:6000]+2.5,linestyle='dotted',color='r',alpha=0.4)
ax.set_xlabel('mili second')
ax.set_ylabel('ms^2/Hz')
ax.get_legend().remove()

abnorm =t_pred.values-df_discord_1[3:6000]+2.5
abnorm = abnorm.dropna()
abnorm=np.array(abnorm)
abnorm.sort(axis=0)
th=abnorm[::-1][20]
np.set_printoptions(threshold=np.inf)
plt.plot([0,6000], [th,th] , color='red', linestyle='-', linewidth=0.5)
plt.show()
"""
