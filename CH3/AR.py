import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import statsmodels
import io
import requests

from statsmodels.tsa.arima_model import ARMA

# 月ごとの飛行機の乗客数データ
url = "https://www.analyticsvidhya.com/wp-content/uploads/2016/02/AirPassengers.csv"
stream = requests.get(url).content
content = pd.read_csv(io.StringIO(stream.decode('utf-8')),index_col='Month',parse_dates=True,dtype='float')

passengers = content['#Passengers'][:120]
passengers_plot= content['#Passengers']
#passengers= np.diff(np.log(passengers))

plt.plot(passengers_plot)



result = sm.tsa.stattools.adfuller(passengers)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))
sm.graphics.tsa.plot_acf(passengers, lags=35)
sm.graphics.tsa.plot_pacf(passengers, lags=35)


ar = sm.tsa.AR(passengers)
print ('the order of arma is', ar.select_order(maxlag=6, ic='aic'))
AR = ARMA(passengers, order=(5, 0)).fit(dist=False)

resid = AR.resid
fig = plt.figure(figsize=(5,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(resid, lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(resid, lags=40, ax=ax2)


pred = AR.predict('1955-01-01', '1958-12-01')
plt.figure(figsize=(6,5))
plt.plot(passengers[70:120],'--')
plt.plot(pred, "k")
plt.xticks(rotation=45)


pred = AR.predict('1958-01-01', '1965-12-01')
plt.figure(figsize=(4,5))
plt.plot(passengers[40:],'--')
plt.plot(pred, "k")
plt.xticks(rotation=45)

plt.show()

"""
pred = AR.predict('1950-01-01', '1953-12-01')
plt.figure(figsize=(6,5))
plt.plot(passengers[10:60],'--')
plt.plot(pred, "k")
plt.xticks(rotation=45)


pred = AR.predict('1958-01-01', '1965-12-01')
plt.figure(figsize=(4,5))
plt.plot(passengers[40:],'--')
plt.plot(pred, "k")
plt.xticks(rotation=45)

plt.show()
"""
