#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import io
import requests
from statsmodels.tsa.arima_model import ARIMA

# 月ごとの飛行機の乗客数データ
url = "https://www.analyticsvidhya.com/wp-content/uploads/2016/02/AirPassengers.csv"
stream = requests.get(url).content
content = pd.read_csv(io.StringIO(stream.decode('utf-8')),index_col='Month',parse_dates=True,dtype='float')

passengers = content['#Passengers'][:120]


sm.graphics.tsa.plot_acf(passengers, lags=40)


sm.graphics.tsa.plot_pacf(passengers, lags=35)

ARIMA = ARIMA(passengers, order=(3, 2, 1)).fit(dist=False)
resid = ARIMA.resid
fig = plt.figure(figsize=(6,9))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(resid.values.squeeze(), lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(resid, lags=40, ax=ax2)



pred = ARIMA.predict('1955-01-01', '1958-12-01',typ='levels')
plt.figure(figsize=(6,5))
plt.plot(passengers[70:120],'--')
plt.plot(pred, "k")
plt.xticks(rotation=45)


pred = ARIMA.predict('1958-01-01', '1965-12-01',typ='levels')
plt.figure(figsize=(4,5))
plt.plot(passengers[40:],'--')
plt.plot(pred, "k")
plt.xticks(rotation=45)

plt.show()
