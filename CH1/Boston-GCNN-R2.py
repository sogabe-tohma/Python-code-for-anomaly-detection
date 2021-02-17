# -*- coding: utf-8 -*-

import sys
sys.path.append('/home/user/Desktop/TKFile/GraphCNN-Origin/code')
#from metric import Distance
from graph_convolution import GraphConv

import math
import numpy  as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.optimizers   import adam, RMSprop
from keras.regularizers import l2, l1

from sklearn.preprocessing   import normalize, StandardScaler
from sklearn.model_selection import train_test_split



###### Setting ###############################################################################################
seed_val = 1984
np.random.seed(seed_val)

sns.set_style("whitegrid", {'axes.edgecolor' : '.4'})
plt.rcParams['axes.linewidth'] = 4

###### Data Transform ############################################################################################################
loc  = 'boston.csv'
data = pd.read_csv(open(loc,'r'))
data_x = data.drop(["MV"], axis = 1).values
data_y = data["MV"].values

X_train, X_test, y_train, y_test = np.array(train_test_split(data_x, data_y, test_size=0.2, random_state=364364))

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

###### feature map ###############################################################################################################
# 相関行列GraphCNN
def correlation(num_neighbors):
    corr_mat  = np.array(normalize(np.abs(np.corrcoef(X_train.transpose())), norm='l1', axis=1),dtype='float64')
    graph_mat = np.argsort(corr_mat,1)[:,-num_neighbors:]
    return graph_mat

# 距離行列GraphCNN
def gaussiankernel(num_neighbors, sigma):
    X_trainT = X_train.T
    row  = X_trainT.shape[0]
    kernel_mat = np.zeros(row * row).reshape(row, row)
    for i in range(row):
        for j in range(row):
            kernel_mat[i, j] = math.exp( - (np.linalg.norm(X_trainT[i] - X_trainT[j]) ** 2) / (2 * sigma ** 2))
    graph_mat  = np.argsort(kernel_mat, 1)[:,-num_neighbors:]
    return graph_mat

def DBSCAN(num_neighbors, num_q):
    X_trainT = X_train.T
    t = Topology()
    t.load_data(X_trainT, standardize=True)
    t.fit_transform(lens=[PCA(components=[0,1])])

    d = Distance(metric="euclidean")
    dist_matrix = d.fit_transform(X_trainT)
    transition_matrix = np.zeros([X_trainT.shape[0], X_trainT.shape[0]])

    q = num_q
    for j in range(X_trainT.shape[0]):
        target_index = j
        dist_from_target = dist_matrix[target_index]
        neighbor_index = (np.argsort(dist_from_target) <= q)
        neighbor_index[target_index] = False
        similarity = 1 / dist_from_target[neighbor_index]
        transition_probability = similarity / np.sum(similarity)
        transition_matrix[target_index, neighbor_index] += transition_probability
        node_index_has_target_data = t._node_index_from_data_id([target_index])

        for i in node_index_has_target_data:
            data_index_in_node = np.array(t.hypercubes[i])
            transition_dist = dist_matrix[target_index, data_index_in_node]
            similarity = 1 / transition_dist
            similarity[np.isinf(similarity)] = 0
            transition_probability = similarity / np.sum(similarity)
            transition_matrix[target_index, data_index_in_node] += transition_probability

        if(len(node_index_has_target_data)):
            transition_matrix[target_index] /= (np.sum(transition_matrix[target_index]))
    graph_mat = np.argsort(transition_matrix, 1)[:,-num_neighbors:]

    return graph_mat

###### Model Construction ########################################################################################################
sigma = 81
num_q = 5
num_neighbors = 12

graph_mat = correlation(num_neighbors)
# graph_mat = gaussiankernel(num_neighbors, sigma)
#graph_mat = DBSCAN(num_neighbors, num_q)

epoch = 330
epochs = np.arange(epoch)
results_train = np.ones(epoch) * 20
results_test  = np.ones(epoch) * 20

batch_size = 30
num_hidden = 32
filters_1  = 20
filters_2  = 22
filters_3  = 22
filters_4  = 16
filters_5  = 16

model = Sequential()
model.add(GraphConv(filters=filters_1, neighbors_ix_mat = graph_mat, num_neighbors=num_neighbors, input_shape=(X_train.shape[1],1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(GraphConv(filters=filters_2, neighbors_ix_mat = graph_mat, num_neighbors=num_neighbors))
model.add(Activation('relu'))
model.add(GraphConv(filters=filters_3, neighbors_ix_mat = graph_mat, num_neighbors=num_neighbors))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(GraphConv(filters=filters_4, neighbors_ix_mat = graph_mat, num_neighbors=num_neighbors))
model.add(Activation('relu'))
model.add(GraphConv(filters=filters_5, neighbors_ix_mat = graph_mat, num_neighbors=num_neighbors))
model.add(Activation('relu'))

model.add(Flatten())
model.add(Dense(num_hidden))
model.add(Activation('relu'))
model.add(Dense(1, kernel_regularizer=l2(0.01)))

model.compile(loss='mean_squared_error', optimizer='adam')

for i in epochs:
    model.fit(X_train.reshape(X_train.shape[0],X_train.shape[1],1), y_train, epochs=1, batch_size=batch_size, verbose=0, validation_split=0.2)

    pred_train = model.predict(X_train.reshape(X_train.shape[0],X_train.shape[1],1), batch_size=5).flatten()
    pred_test  = model.predict(X_test.reshape(X_test.shape[0],X_test.shape[1],1), batch_size=5).flatten()

    R_train = (np.corrcoef(y_train, pred_train)**2)[0,1]
    R_test  = (np.corrcoef(y_test,  pred_test)**2)[0,1]
    results_train[i] = R_train
    results_test[i]  = R_test
    print('Epoch: %d, R_train: %.3f, R_test: %.3f'%(i, R_train, R_test))

###### Figure ##########################################################################################################
print('-----------------------------------------------------')
print('Max_R2_train:  ', results_train.max())
print('Max_R2_test:   ', results_test.max())
print('Mean_R2_train: ', results_train[150:200].mean())
print('Mean_R2_test:  ', results_test[150:200].mean())

num  = 30
num2 = num//2 + 1
smooth  = np.ones(num)/num
smooth1 = np.convolve(results_train, smooth, mode='same')
smooth2 = np.convolve(results_test,  smooth, mode='same')

plt.figure(figsize=(12, 8))
plt.plot(epochs, results_train, color='cornflowerblue', linestyle='None', marker='.', markersize=3, label='Train')
plt.plot(epochs, results_test,  color='orange', linestyle='None', marker='.', markersize=3, label='Test')
plt.plot(epochs[num2:epoch-num2,], smooth1[num2:epoch-num2,], color='cornflowerblue', linewidth=3, label='Train smooth')
plt.plot(epochs[num2:epoch-num2,], smooth2[num2:epoch-num2,], color='orange', linewidth=3, label='Test smooth')
plt.xlabel('Epoch', fontsize=16)
plt.ylabel('R^2',  fontsize=16)
plt.ylim(0.8, 1)
plt.tick_params(labelsize=14)
plt.legend(fontsize=18, loc='lower right')
plt.show()

np.savetxt('Boston-C-GCNN-train-R^2.csv', results_train, delimiter=',')
np.savetxt('Boston-C-GCNN-test-R^2.csv',  results_test,  delimiter=',')
np.savetxt('Boston-C-GCNN-train-R^2-smooth.csv', smooth1,  delimiter=',')
np.savetxt('Boston-C-GCNN-test-R^2-smooth.csv',  smooth2,  delimiter=',')

np.savetxt('Boston-K-GCNN-train-R^2.csv', results_train, delimiter=',')
np.savetxt('Boston-K-GCNN-test-R^2.csv',  results_test,  delimiter=',')
np.savetxt('Boston-K-GCNN-train-R^2-smooth.csv', smooth1,  delimiter=',')
np.savetxt('Boston-K-GCNN-test-R^2-smooth.csv',  smooth2,  delimiter=',')

np.savetxt('Boston-D-GCNN-train-R^2.csv', results_train, delimiter=',')
np.savetxt('Boston-D-GCNN-test-R^2.csv',  results_test,  delimiter=',')
np.savetxt('Boston-D-GCNN-train-R^2-smooth.csv', smooth1,  delimiter=',')
np.savetxt('Boston-D-GCNN-test-R^2-smooth.csv',  smooth2,  delimiter=',')

###### Compare ########################################################################################################################
loc1 = '/home/user/Desktop/CNN/CSV/Boston-NN-test-R^2-smooth.csv'
loc2 = '/home/user/Desktop/CNN/CSV/Boston-C-GCNN-test-R^2-smooth.csv'
loc3 = '/home/user/Desktop/CNN/CSV/Boston-K-GCNN-test-R^2-smooth.csv'
loc4 = '/home/user/Desktop/CNN/CSV/Boston-D-GCNN-test-R^2-smooth.csv'
NN = pd.read_csv(open(loc1,'r'))
CGCNN = pd.read_csv(open(loc2,'r'))
KGCNN = pd.read_csv(open(loc3,'r'))
DGCNN = pd.read_csv(open(loc4,'r'))

plt.figure(figsize=(12, 16))
plt.subplot(2, 1, 1)
plt.plot(NN[:300],      color='grey',   linewidth=4, label='NN')
plt.plot(CGCNN[:300],   color='cornflowerblue', linewidth=4, label='C-GCNN')
plt.plot(KGCNN[:300],   color='green',  linewidth=4, label='K-GCNN')
plt.plot(DGCNN[:300],   color='red',    linewidth=4, label='D-GCNN')
plt.plot(smooth2[:300], color='orange', linewidth=4, label='New-GCNN')
plt.xlabel('Epoch', fontsize=16)
plt.ylabel('R^2',   fontsize=16)
plt.ylim(0, 1)
plt.tick_params(labelsize=14)
plt.legend(fontsize=18, loc='lower right')

plt.subplot(2, 1, 2)
plt.plot(NN[:300],      color='grey',   linewidth=4, label='NN')
plt.plot(CGCNN[:300],   color='cornflowerblue', linewidth=4, label='C-GCNN')
plt.plot(KGCNN[:300],   color='green',  linewidth=4, label='K-GCNN')
plt.plot(DGCNN[:300],   color='red',    linewidth=4, label='D-GCNN')
plt.plot(smooth2[:300], color='orange', linewidth=4, label='New-GCNN')
plt.xlabel('Epoch', fontsize=16)
plt.ylabel('R^2',   fontsize=16)
plt.ylim(0.8, 1)
plt.tick_params(labelsize=14)
plt.legend(fontsize=18, loc='lower right')

plt.show()
