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

result = sm.tsa.stattools.adfuller(df_nile['volume'].iloc[4001:])
#result = sm.tsa.stattools.adfuller(np.diff(np.log(passengers)))




df_test = df_nile['volume'].iloc[4000:6001]
df_train = df_nile['volume'].iloc[:4000]

fig = plt.figure(figsize=(12,6))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(df_train.values, lags=600, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(df_train.values, lags=600,  ax=ax2)
plt.show()


# from statsmodels.tsa import stattools as st
# ARMAモデルの次数を決める

arma_11 = sm.tsa.ARMA(df_train, (5, 2)).fit()

arma_11_inpred = arma_11.predict(start=2, end=4000,typ='levels')
# out-of-sample predict
arma_11_outpred = arma_11.predict(start=3999, end=6000,typ='levels')
# plot data and predicted values




def plot_ARMA_results(origdata, pred11in, pred11out):
    ax = origdata['volume'].plot(figsize=(10,1), grid=False,
    	color='k',marker ='o', markersize=2,markerfacecolor='w')
    pred11in.plot(color=['b'],linestyle='dotted')
    pred11in1= pred11in-df_train+3.0
    pred11out1 = pred11out-df_test+3.0
    pred11out.plot(color=['r'])
    pred11out1.plot(color=['r'],linestyle='dotted')
    pred11in1.plot(color=['b'],linestyle='dotted')
    ax.set_xlabel('mili second')
    ax.set_ylabel('ms^2/Hz')
    ax.set_ylim(1,7)
    plt.show()
#call the plot
plot_ARMA_results(df_nile, arma_11_inpred, arma_11_outpred)
