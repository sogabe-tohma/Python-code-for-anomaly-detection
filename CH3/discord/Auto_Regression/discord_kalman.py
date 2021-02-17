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

df_test = df_nile['volume'].iloc[4000:6001]
df_train = df_nile['volume'].iloc[2:4000]

# Unobserved Components Modeling (via Kalman Filter)の実行
import statsmodels.api as sm

# Fit a local level model
mod_ll = sm.tsa.UnobservedComponents(df_train, 'local level')
res_ll = mod_ll.fit()
pred=res_ll.predict(start=3999, end=6000,typ='levels')

res_11_1=res_ll.fittedvalues[2:]-df_train

#print ('resis', res_ll.fittedvalues[:])

# Show a plot of the estimated level and trend component series
#fig_ll = res_ll.plot_components(legend_loc="upper left", figsize=(12,8))

plt.plot(df_train,'k')
plt.plot(res_ll.fittedvalues[2:],'b',linestyle='dotted')
plt.plot(res_11_1,'b',linestyle='dotted')
plt.plot(df_test,'k')
plt.plot(pred,'r')
plt.plot(pred-df_test,'r',linestyle='dotted')
plt.show()
