#!/usr/bin/env python
# coding: utf-8


import numpy as np
import numpy.random as rd
import pandas as pd
import matplotlib
from matplotlib import font_manager
import matplotlib.pyplot as plt
import seaborn as sns
import io
import requests
import statsmodels.api as sm

sns.set(style="whitegrid", palette="muted", color_codes=True)
# 月ごとの飛行機の乗客数データ
url = "https://www.analyticsvidhya.com/wp-content/uploads/2016/02/AirPassengers.csv"
stream = requests.get(url).content
content = pd.read_csv(io.StringIO(stream.decode('utf-8')),index_col='Month',parse_dates=True,dtype='float')

passengers = content['#Passengers'][:120]

passengers[80]=  passengers[80]*1

passengers_plot= content['#Passengers']


#model = sm.tsa.UnobservedComponents(passengers, 'local level')
#model = sm.tsa.UnobservedComponents(passengers, 'local linear trend')
#model = sm.tsa.UnobservedComponents(passengers,'local linear trend',seasonal=12)
#model = sm.tsa.UnobservedComponents(passengers, 'random walk with drift',seasonal=12)
model = sm.tsa.UnobservedComponents(passengers,'local linear deterministic trend',
	seasonal=12)
kalman = model.fit(method='bfgs')
residkalman= kalman.resid
fig = plt.figure(figsize=(6,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(residkalman.values.squeeze(), lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(residkalman, lags=40, ax=ax2)


pred = kalman.predict('1955-01-01', '1958-12-01',typ='levels')
plt.figure(figsize=(6,5))
plt.plot(passengers[70:120],'--')
plt.plot(pred, "k")
plt.xticks(rotation=45)

pred = kalman.predict('1958-01-01', '1965-12-01',typ='levels')
plt.figure(figsize=(4,5))
plt.plot(passengers[40:],'--')
plt.plot(pred, "k")
plt.xticks(rotation=45)

plt.show()
