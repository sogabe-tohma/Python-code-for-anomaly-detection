# -*- coding: utf-8 -*-

import sys
sys.path.append('/home/user/Desktop/TKFile/GraphCNN-Origin/code')
from graph_convolution import GraphConv

import math
import time
import numpy  as np
import pandas as pd
import statistics

import seaborn as sns
import matplotlib.pyplot as plt

from keras.models       import Sequential, model_from_json
from keras.layers       import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.callbacks    import EarlyStopping
from keras.optimizers   import adam, RMSprop, SGD, Adagrad, Adadelta
from keras.regularizers import l2, l1

from sklearn.metrics         import mean_squared_error
from sklearn.preprocessing   import normalize
from sklearn.model_selection import train_test_split

###### Setting ####################################################################################################
seed_val = 1984
np.random.seed(seed_val)

point = []
start = time.time()
plt.figure(figsize = (12, 8))
sns.set_style("whitegrid")

###### Data Transform #############################################################################################
am_loc1 = '/home/user/Desktop/GridWorks/Dataset/train_data_1_2.csv'
am_loc2 = '/home/user/Desktop/GridWorks/Dataset/test_data_1_2.csv'
data_x  = np.array(pd.read_csv(open(am_loc1,'r')))
data_y  = np.array(pd.read_csv(open(am_loc2,'r')))

# data_x_max = data_x.max(0),
# data_y_max = data_y.max(0)
# data_x = data_x / data_x_max
# data_y = data_y / data_y_max

X_train, y_train = data_x[:,1:], data_x[:,0]
X_test,  y_test  = data_y[:,1:], data_y[:,0]

###### Correlation ################################################################################################
num_neighbors = 2
corr_mat  = np.array(normalize(np.abs(np.corrcoef(X_train.transpose())), norm='l1', axis=1), dtype='float64')
graph_mat = np.argsort(corr_mat,1)[:,-num_neighbors:]

###### GaussianKernel #############################################################################################
# X_trainT = X_train.T
# row  = X_trainT.shape[0]
# kernel_mat = np.zeros(row * row).reshape(row, row)

# sigma = 1
# num_neighbors = 6
# for i in range(row):
#     for j in range(row):
#         kernel_mat[i, j] = math.exp( - (np.linalg.norm(X_trainT[i] - X_trainT[j]) ** 2) / (2 * sigma ** 2))
# graph_mat  = np.argsort(kernel_mat, 1)[:,-num_neighbors:]

###### Learning ###################################################################################################
epoch  = 800
epochs = np.arange(epoch)
results_train = np.ones(epoch) * 20
results_test  = np.ones(epoch) * 20

batch_size = 50
num_hidden = 75
filters_1  = 24
filters_2  = 24
filters_3  = 21
filters_4  = 22
filters_5  = 20

model = Sequential()
model.add(GraphConv(filters=filters_1, neighbors_ix_mat = graph_mat, num_neighbors=num_neighbors, activation='relu', input_shape=(X_train.shape[1],1)))
model.add(BatchNormalization())
# model.add(Dropout(0.25))
model.add(GraphConv(filters=filters_2, neighbors_ix_mat = graph_mat, num_neighbors=num_neighbors, activation='relu'))
model.add(BatchNormalization())
# model.add(Dropout(0.25))
model.add(GraphConv(filters=filters_3, neighbors_ix_mat = graph_mat, num_neighbors=num_neighbors, activation='relu'))
# model.add(BatchNormalization())
# model.add(Dropout(0.25))
model.add(GraphConv(filters=filters_4, neighbors_ix_mat = graph_mat, num_neighbors=num_neighbors, activation='relu'))
# model.add(BatchNormalization())
# model.add(Dropout(0.25))
model.add(GraphConv(filters=filters_5, neighbors_ix_mat = graph_mat, num_neighbors=num_neighbors, activation='relu'))
model.add(BatchNormalization())
# model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(num_hidden, kernel_regularizer=l2(0.01),))
model.add(BatchNormalization())
model.add(Activation('relu'))
# model.add(Dropout(0.1))
model.add(Dense(1, kernel_regularizer=l2(0.01)))
# model.add(Dropout(0.1))
model.summary()
model.compile(loss='mean_squared_error', optimizer='adam')

for i in epochs:
    ramdom1 = np.random.choice(len(data_x), len(data_x)//2, replace = False)
    ramdom2 = np.random.choice(len(data_y), len(data_y)//2, replace = False)
    data_xb = data_x[ramdom1]
    data_yb = data_y[ramdom2]
    X_train, y_train = data_xb[:,1:], data_xb[:,0]
    X_test,  y_test  = data_yb[:,1:], data_yb[:,0]

    model.fit(X_train.reshape(X_train.shape[0],X_train.shape[1],1), y_train, epochs=1, batch_size=batch_size,
                              shuffle=True,validation_split=0.25, verbose = 0,)

    pred_train = np.array(model.predict(X_train.reshape(X_train.shape[0],X_train.shape[1],1), batch_size = 10)).flatten()
    pred_test  = np.array(model.predict(X_test.reshape(X_test.shape[0],X_test.shape[1],1), batch_size = 10)).flatten()
    RMSE_train = np.sqrt(mean_squared_error(y_train, pred_train))
    RMSE_test  = np.sqrt(mean_squared_error(y_test,  pred_test))
    # RMSE_train = np.sqrt(mean_squared_error(y_train * 80, pred_train * 80))
    # RMSE_test  = np.sqrt(mean_squared_error(y_test  * 80, pred_test  * 80))

    results_train[i] = RMSE_train
    results_test[i]  = RMSE_test
    Min_RMSE_test    = results_test.min()

    if  RMSE_test == Min_RMSE_test:
        print('Epoch: %d, Train_RMSE: %.4f, Min_RMSE_test: %4f'%(i, RMSE_train, RMSE_test))
    else:
        print('Epoch: %d, Train_RMSE: %.4f, RMSE_test: %4f'    %(i, RMSE_train, RMSE_test))

    ###### RealTime RMSE Plot ######
    # plt.plot(epochs, results_train, color='blue',  linestyle='--', )
    # plt.plot(epochs, results_test,  color='green', linestyle='--', )
    # plt.ylim(4, 12)
    # plt.pause(0.005)
    # plt.cla()

###### Results #########################################################################################################
pred_train = np.array(model.predict(X_train.reshape(X_train.shape[0],X_train.shape[1],1), batch_size=5)).flatten()
pred_test  = np.array(model.predict(X_test.reshape(X_test.shape[0],X_test.shape[1],1), batch_size=5)).flatten()
RMSE_train = np.sqrt(mean_squared_error(y_train, pred_train))
RMSE_test  = np.sqrt(mean_squared_error(y_test,  pred_test))
process_time = time.time() - start

print(np.round(y_test[-10:,]))
print(np.round(pred_test[-10:,]))
print('Min_Train-RMSE: ', results_train.min())
print('Min_Test-RMSE:  ', results_test.min())
print('process_time:   ', process_time)

###### RMSE Comparing Figure ###########################################################################################
num = 20
smooth  = np.ones(num)/num
smooth1 = np.convolve(results_train, smooth, mode='same')
smooth2 = np.convolve(results_test,  smooth, mode='same')

plt.close()
plt.figure(figsize=(12, 9))
plt.plot(epochs, results_train, color='blue',  linestyle='None', marker='.', markersize=1, label='Train')
plt.plot(epochs, results_test,  color='green', linestyle='None', marker='.', markersize=1, label='Test',)
plt.plot(epochs[11:epoch-11,], smooth1[11:epoch-11,], color='blue' , linewidth=3, label='Train smooth')
plt.plot(epochs[11:epoch-11,], smooth2[11:epoch-11,], color='green', linewidth=3, label='Test smooth')
plt.xlabel('Epoch', fontsize=16)
plt.ylabel('RMSE'  , fontsize=16)
plt.ylim(0, 15)
plt.legend(fontsize=15)
plt.show()
