#!/usr/bin/env python
# coding: utf-8

import numpy
import matplotlib.pyplot as plt
import pandas as pd
import math
from keras.models import Sequential
from keras.layers import Dense

import io
import requests

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)


url = "https://www.analyticsvidhya.com/wp-content/uploads/2016/02/AirPassengers.csv"
stream = requests.get(url).content
#content = pd.read_csv(io.StringIO(stream.decode('utf-8')),index_col='Month',parse_dates=True,dtype='float')
content = pd.read_csv(io.StringIO(stream.decode('utf-8')), usecols=[1], engine='python', skipfooter=3)

dataset = content.values



# split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
# reshape dataset
look_back = 4
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)


from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.model_selection import train_test_split, GridSearchCV

regressor1 = LinearRegression()
regressor2 =  SVR(kernel='linear', C=1e3)
regressor3 = RFR(n_jobs=-1, random_state=2525)
#regressor1.fit(trainX,trainY)
regressor2.fit(trainX,trainY)
#regressor3.fit(trainX,trainY)

# 学習結果を出力する
plt.figure(figsize=(6,3))
plt.plot(trainY, "--", color = 'b')
#plt.plot(regressor1.predict(trainX), color = 'k')
plt.plot(regressor2.predict(trainX), color = 'k')
#plt.plot(regressor3.predict(trainX), color = 'k')

# 予測結果を出力する
plt.figure(figsize=(4,3))
plt.plot(testY, "--", color = 'b')
#plt.plot(regressor1.predict(testX), color = 'k')
plt.plot(regressor2.predict(testX), color = 'k')
#plt.plot(regressor3.predict(testX), color = 'k')

plt.show()
