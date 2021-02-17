from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

import statsmodels.graphics.api as smg
from statsmodels.graphics.tsaplots import plot_pacf


from statsmodels.tsa.ar_model import AR
from statsmodels.tsa import stattools as st
import statsmodels.api as sm

import matplotlib.pyplot as plt
import numpy as np

# convert series to supervised learning
def series_to_supervised(data, n_in=2, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

# load dataset
dataset = read_csv('pollution_5variable.csv', header=0, index_col=0)
values = dataset.values
dataset = dataset['pollution']

"""
dataset1 = dataset['pollution']
dataset =  dataset['pollution'][0:280]

print(st.arma_order_select_ic(dataset, ic='bic', trend = 'nc'))

arma_10 = sm.tsa.ARMA(dataset, (1, 0)).fit()

arma_10_inpred = arma_10.predict(start = 0, end =280)

print ('testprint', arma_10_inpred  )
# out-of-sample predict
arma_10_outpred = arma_10.predict(start= 250, end = 380)


plt.figure(figsize=(8,4))

plt.plot(dataset1,'--')

plt.plot(arma_10_inpred, "k")

plt.plot(arma_10_outpred, "r")

plt.xticks(rotation=45)






# 残差のチェック
residSARIMA = arma_10.resid

sm.graphics.tsa.plot_acf(residSARIMA, lags=40)
sm.graphics.tsa.plot_pacf(residSARIMA, lags=40)

plt.show()


"""

dataset = read_csv('pollution_5variable.csv', header=0, index_col=0)
values = dataset.values
dataset = dataset['pollution']

result = sm.tsa.stattools.adfuller(dataset)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))

sm.graphics.tsa.plot_acf(dataset, lags=80,)
sm.graphics.tsa.plot_pacf(dataset, lags=80,)
plt.show()


"""
# ACF, PACF
fig = plt.figure(figsize=(12,3))
ax1 = fig.add_subplot(121)
fig = sm.graphics.tsa.plot_acf(dataset, lags=80, markersize=2, ax=ax1)
ax2 = fig.add_subplot(122)
fig = sm.graphics.tsa.plot_pacf(dataset, lags=80, markersize=2,  ax=ax2)
plt.show()
"""




# integer encode direction
encoder = LabelEncoder()
values[:,4] = encoder.fit_transform(values[:,4])
# ensure all data is float
values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
# frame as supervised learning
reframed = series_to_supervised(scaled, 1, 1)
# drop columns we don't want to predict
reframed.drop(reframed.columns[[9,10,11,12,13,14,15]], axis=1, inplace=True)
print(reframed.head())





# split into train and test sets
values = reframed.values
n_train_hours =10*24
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]
# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# design network
model = Sequential()
model.add(LSTM(200, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(train_X, train_y, epochs=200, batch_size=36, validation_data=(test_X, test_y), verbose=2, shuffle=False)
# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()


pm_pred = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
playback_pm_pred = concatenate((pm_pred, test_X[:, 1:]), axis=1)
playback_pm_pred = scaler.inverse_transform(playback_pm_pred)
playback_pm_pred = playback_pm_pred[:,0]


test_pm = test_y.reshape((len(test_y), 1))
playback_pm = concatenate((test_pm, test_X[:, 1:]), axis=1)
playback_pm = scaler.inverse_transform(playback_pm)
playback_pm =playback_pm[:,0]

rmse = sqrt(mean_squared_error(playback_pm, playback_pm_pred))
print('Test RMSE: %.3f' % rmse)
# ACF, PACF
dataset =  np.abs(playback_pm-playback_pm_pred)
fig = plt.figure(figsize=(12,3))
ax1 = fig.add_subplot(121)
fig = sm.graphics.tsa.plot_acf(dataset, lags=80, markersize=2, ax=ax1)
ax2 = fig.add_subplot(122)
fig = sm.graphics.tsa.plot_pacf(dataset, lags=80, markersize=2,  ax=ax2)
pyplot.show()


pyplot.plot(playback_pm,linestyle= '--', label='actual')
pyplot.plot(playback_pm_pred,label='predict')
pyplot.legend()
pyplot.show()
