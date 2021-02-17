#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import io
import requests

# 月ごとの飛行機の乗客数データ
url = "https://www.analyticsvidhya.com/wp-content/uploads/2016/02/AirPassengers.csv"
stream = requests.get(url).content
content = pd.read_csv(io.StringIO(stream.decode('utf-8')),index_col='Month',parse_dates=True,dtype='float')

passengers = content['#Passengers'][:120]

passengers_plot= content['#Passengers']

#passengers_diff= passengers .diff()[:120].dropna()

#plt.plot(passengers_diff)

#ctt = sm.tsa.stattools.adfuller(np.diff(np.log(passengers)), regression = "ctt")
#ct = sm.tsa.stattools.adfuller(np.diff(np.log(passengers)), regression = "ct")
#c = sm.tsa.stattools.adfuller(np.diff(np.log(passengers)), regression = "c")


MA_order= sm.tsa.arma_order_select_ic(passengers, max_ar=0, max_ma=4, ic=['aic','bic'])
print ('order is', MA_order)

MA = sm.tsa.ARMA(passengers, order=(0, 1)).fit()
resid = MA.resid
fig = plt.figure(figsize=(5,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(resid, lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(resid, lags=40, ax=ax2)

pred = MA.predict('1955-01-01', '1958-12-01')
plt.figure(figsize=(6,5))
plt.plot(passengers[70:120],'--')
plt.plot(pred, "k")
plt.xticks(rotation=45)


pred = MA.predict('1958-01-01', '1965-12-01')
plt.figure(figsize=(4,5))
plt.plot(passengers[40:],'--')
plt.plot(pred, "k")
plt.xticks(rotation=45)

plt.show()
