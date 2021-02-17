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
plt.plot(passengers_plot)
plt.show()

#ctt = sm.tsa.stattools.adfuller(np.diff(np.log(passengers)), regression = "ctt")
#ct = sm.tsa.stattools.adfuller(np.diff(np.log(passengers)), regression = "ct")
#c = sm.tsa.stattools.adfuller(np.diff(np.log(passengers)), regression = "c")
#result = sm.tsa.stattools.adfuller(np.diff(np.log(passengers)))

result = sm.tsa.stattools.adfuller(passengers)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))
sm.graphics.tsa.plot_acf(passengers, lags=35)
sm.graphics.tsa.plot_pacf(passengers, lags=35) 
plt.show()
