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


# Data Split (70: 30)


df_test = df_nile['volume'].iloc[4000:6001]
df_train = df_nile['volume'].iloc[:4000]
df_train_all = df_nile['volume'].iloc[:6000]

arma_11 = sm.tsa.ARMA(df_train, (3, 1)).fit()

arma_11_all = sm.tsa.ARMA(df_train_all, (3, 1)).fit()

arma_11_inpred = arma_11.predict(start=2, end=4000,typ='levels')
# out-of-sample predict
arma_11_outpred = arma_11.predict(start=3999, end=6000,typ='levels')
# plot data and predicted values
arma_11_pred_all = arma_11_all.predict(typ='levels')
# plot data and predicted values



fig = plt.figure(figsize=(12,6))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(arma_11_all.resid, lags=600, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(arma_11_all.resid, lags=600,  ax=ax2)
plt.show()




def plot_ARMA_results(origdata, pred11):
    ax = origdata['volume'].plot(figsize=(10,2), grid=False,color='k',marker ='o', markersize=2,markerfacecolor='w')
    pred11.plot(color=['b'],linestyle='dotted')
    pred11in1= (df_train_all-pred11)+3.0

    pred11in1.plot(color=['b'],linestyle='dotted')

    ax.set_xlabel('mili second')
    ax.set_ylabel('ms^2/Hz')
    ax.set_ylim(2,7)


#call the plot

plot_ARMA_results(df_nile, arma_11_pred_all)
th=2.38549748e-01+3
plt.plot([0,6000], [th,th] , color='red', linestyle='-', linewidth=0.5)

plt.show()

abnormalty_sort = np.sort(df_train_all-arma_11_pred_all)
np.set_printoptions(threshold=np.inf)
print ('a is ', abnormalty_sort[::-1] )
