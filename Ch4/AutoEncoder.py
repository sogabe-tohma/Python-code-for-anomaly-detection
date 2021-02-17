import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

import sys
sys.setrecursionlimit(10000)
import gc

import renom as rm
from renom.optimizer import Adam
from renom.cuda import set_cuda_active
set_cuda_active(False) # set True for larger dataset



# 実験1 人工データ
x = np.arange(200)
normal_data = np.sin(x*0.5).reshape(200, -1)
abnormal_data = (np.sin(x*0.5)**2).reshape(200, -1)

whole_data = np.vstack((normal_data, abnormal_data))
whole_data = np.vstack((whole_data, normal_data))

plt.figure(figsize=(15,5))
plt.tick_params(top=1, right=1, direction='in')
plt.plot(whole_data, color='black')
plt.ylabel('value')
plt.xlabel('time')
plt.show()



# データの前処理
L = 5 # length of series data
def create_subseq(ts, stride):
    sub_seq = []
    for i in range(0, len(ts), stride):
        if len(ts[i:i+L]) == L:
            sub_seq.append(ts[i:i+L])
    return sub_seq
sub_seq = create_subseq(normal_data, 2)
X_train, X_test = train_test_split(sub_seq, test_size=0.2)
X_train, X_test = np.array(X_train).astype(np.float32),np.array(X_test).astype(np.float32)
train_size, test_size = X_train.shape[0], X_test.shape[0]
print('train size:{}, test size:{}'.format(train_size, test_size))



# 学習
m = 1 # dim of series data
c = 2 # dim of hidden layer of lstm

class EncDecAD(rm.Model):
    def __init__(self):
        super(EncDecAD, self).__init__()
        self.encoder = rm.Lstm(c)
        self.decoder = rm.Lstm(c)
        self.linear      = rm.Dense(m)


    def truncate(self):
        self.encoder.truncate()
        self.decoder.truncate()


    def forward(self, input_seq, train=False):
        loss = 0
        reconst_seq= []

        # encoding phase
        for t in range(input_seq.shape[1]):
            hE = self.encoder(input_seq[:, t])

        # Set the initial state of the decoder's LSTM to the final state of the encoder's LSTM.
        self.decoder._z = hE
        self.decoder._state = self.encoder._state

        # reconstruction phase
        # Note: reconstruction is done in reverse order
        reconst = self.linear(hE)
        reconst_seq.append(reconst.as_ndarray())
        loss += rm.mse(reconst, input_seq[:, -1])

        for t in range(1, input_seq.shape[1]):
            hD = self.decoder(input_seq[:, -t]) if train else self.decoder(reconst)
            reconst = self.linear(hD)
            reconst_seq.append(reconst.as_ndarray())
            loss += rm.mse(reconst, input_seq[:, -(t+1)])

        reconst_seq = reconst_seq[::-1]
        reconst_seq = np.transpose(reconst_seq, (1, 0, 2)) # (time_index, batch, value) => (batch, time_index, value)
        return loss, reconst_seq

batch_size = 16
max_epoch = 2000

optimizer = Adam()
enc_dec = EncDecAD()
#Autoencoder学習
epoch = 0
learning_curve, test_curve = [], []
while(epoch < max_epoch):
    epoch += 1
    perm = np.random.permutation(train_size)
    train_loss = 0
    for i in range(train_size // batch_size):
        train_data = X_train[perm[i*batch_size : (i+1)*batch_size]]
      # Forward propagation
        with enc_dec.train():
            loss, _ = enc_dec.forward(train_data, train=True)
        enc_dec.truncate()
        loss.grad().update(optimizer)
        train_loss += loss.as_ndarray()

    train_loss /= (train_size // batch_size)
    learning_curve.append(train_loss)

    # test
    test_loss = 0
    for i in range(test_size // batch_size):
        test_data = X_test[i*batch_size : (i+1)*batch_size]
        loss, reconst_seq = enc_dec.forward(test_data, train=False)
        enc_dec.truncate()
        test_loss += loss.as_ndarray()

    test_loss /= (test_size // batch_size)
    test_curve.append(test_loss)

    # showing loss and reconstructed series data
    if epoch%500 == 0:
        print("epoch : {}, train loss : {}, test loss : {}".format(epoch, train_loss, test_loss))

        fig, axes = plt.subplots(ncols=3, figsize=(12, 5))
        axes[0].plot(test_data[0, :], label='data')
        axes[0].plot(reconst_seq[0, :], label='reconst')
        axes[0].tick_params(top=1, right=1, direction='in')
        axes[1].plot(test_data[1, :])
        axes[1].plot(reconst_seq[1, :])
        axes[1].tick_params(top=1, right=1, direction='in')
        axes[2].plot(test_data[2, :])
        axes[2].plot(reconst_seq[2, :])
        axes[2].tick_params(top=1, right=1, direction='in')
        axes[0].legend()
        plt.show()

plt.figure(figsize=(10,5))
plt.plot(learning_curve, color='black', label='learning curve')
plt.plot(test_curve, color='red', label='test curve')
#plt.title("Learning curve")
plt.xlabel('epoch')
plt.ylabel('loss')
plt.tick_params(top=1, right=1, direction='in')
plt.legend()
plt.show()


# _, train_pred = enc_dec.forward(data, train=False)
# 
# print(train_pred.ravel().shape)
#
# plt.figure(figsize=(10,5))
# plt.plot(data.ravel(), color='black', label='row data')
# plt.plot(train_pred.ravel(), color='cyan', label='pred data')
# plt.tick_params(top=1, right=1, direction='in')
# plt.legend()
# plt.ylabel('value')
# plt.xlabel('time')
# plt.xlim([50, 150])
# plt.show()



# 異常度の定義と可視化
# computing errors
_, reconst_seq = enc_dec.forward(X_test, train=False)
enc_dec.truncate()
errors = np.abs(X_test - reconst_seq).flatten()

# estimation
mean = sum(errors)/len(errors)

cov = 0
for e in errors:
    cov += (e - mean)**2
cov /= len(errors)

print('mean : ', mean)
print('cov : ', cov)


def Mahala_distantce(x,mean,cov):
    return (x - mean)**2 / cov

mahala_dist = []
for e in errors:
    mahala_dist.append(Mahala_distantce(e, mean, cov))


mahala_dist_sort = sorted(np.array(mahala_dist).ravel(), reverse=True)
print(mahala_dist_sort[17])

# calculate Mahalanobis distance
def Mahala_distantce(x,mean,cov):
    return (x - mean)**2 / cov

# anomaly detection
data = create_subseq(whole_data, L)
data = np.array(data).astype(np.float32)


_, reconst_seq = enc_dec.forward(data, train=False)
errors = np.abs(data - reconst_seq)
enc_dec.truncate()

mahala_dist = []
for e in errors:
    mahala_dist.append(Mahala_distantce(e, mean, cov))

fig, axes = plt.subplots(nrows=3, figsize=(15,10))
axes[0].plot(data.ravel(), color='black')
axes[1].plot(reconst_seq.ravel(), color='green')
axes[2].plot(np.array(mahala_dist).ravel(), color='red',label='Mahalanobis Distance')

#axes[0].set_title('Original data', fontsize=20)
#axes[1].set_title('Reconstructed data', fontsize=20)
#axes[2].set_title('Anomaly score', fontsize=20)

#axes[0].grid()
#axes[1].grid()
#axes[2].grid()

axes[0].set_ylabel('value')
axes[1].set_ylabel('value')
axes[2].set_ylabel('Mahalanobis Distance')

axes[0].set_xlabel('time')
axes[1].set_xlabel('time')
axes[2].set_xlabel('time')

axes[0].tick_params(top=1, right=1, direction='in')
axes[1].tick_params(top=1, right=1, direction='in')
axes[2].tick_params(top=1, right=1, direction='in')

th = 5256.8384
axes[2].plot([0,600], [th,th] , color='black', linestyle='-', label='threshold', linewidth=1)


plt.tight_layout()
plt.legend()
plt.show()






# 実験2 心電図データ
# loading ECG data
df = pd.read_csv('qtdbsel102.txt', header=None, delimiter='\t')
ecg = df.iloc[:,2].values
ecg = ecg.reshape(len(ecg), -1)
print('total length of ECG data : ', len(ecg))

plt.figure(figsize=(15,5))
plt.xlabel('time')
plt.ylabel('ECG\'s value')
plt.plot(np.arange(5000), ecg[:5000], color='black')
plt.tick_params(top=1, right=1, direction='in')

#x = np.arange(4200,4400)
#y1 = [np.min(ecg[:5000])]*len(x)
#y2 = [np.max(ecg[:5000])]*len(x)

#plt.fill_between(x, y1, y2, facecolor='g', alpha=.3)
plt.show()



# データの前処理
normal_cycle = ecg[5000:]

plt.figure(figsize=(10,5))
#plt.title("training data")
plt.xlabel('time')
plt.ylabel('ECG\'s value')
plt.tick_params(top=1, right=1, direction='in')
plt.plot(np.arange(5000,8000), normal_cycle[:3000], color='black')# stop plot at 8000 times for friendly visual
plt.show()


L = 50 # length of series data

sub_seq = create_subseq(normal_cycle, 10)

X_train, X_test = train_test_split(sub_seq, test_size=0.1)
X_train, X_test = np.array(X_train).astype(np.float32), np.array(X_test).astype(np.float32)

del df, sub_seq
gc.collect()

train_size, test_size = X_train.shape[0], X_test.shape[0]
print('train size:{}, test size:{}'.format(train_size, test_size))



# 学習
m = 1 # dim of series data
c = 10 # dim of hidden layer of lstm

batch_size = 32
max_epoch = 50
optimizer = Adam()

# Train Loop
enc_dec = EncDecAD()
epoch = 0
learning_curve, test_curve = [], []

while(epoch < max_epoch):
    epoch += 1
    perm = np.random.permutation(train_size)
    train_loss = 0

    for i in range(train_size // batch_size):
        train_data = X_train[perm[i*batch_size : (i+1)*batch_size]]

      # Forward propagation
        with enc_dec.train():
            loss, _ = enc_dec.forward(train_data, train=True)
        enc_dec.truncate()
        loss.grad().update(optimizer)
        train_loss += loss.as_ndarray()

    train_loss /= (train_size // batch_size)
    learning_curve.append(train_loss)

    # test
    test_loss = 0
    for i in range(test_size // batch_size):
        test_data = X_test[i*batch_size : (i+1)*batch_size]
        loss, reconst_seq = enc_dec.forward(test_data, train=False)
        enc_dec.truncate()
        test_loss += loss.as_ndarray()

    test_loss /= (test_size // batch_size)
    test_curve.append(test_loss)

    # showing loss and reconstructed series data
    if epoch%10 == 0:
        print("epoch : {}, train loss : {}, test loss : {}".format(epoch, train_loss, test_loss))

        fig, axes = plt.subplots(ncols=3, figsize=(12, 5))
        axes[0].plot(test_data[0, :], label='data')
        axes[0].plot(reconst_seq[0, :], label='reconst')
        axes[1].plot(test_data[1, :])
        axes[1].plot(reconst_seq[1, :])
        axes[2].plot(test_data[2, :])
        axes[2].plot(reconst_seq[2, :])
        axes[0].tick_params(top=1, right=1, direction='in')
        axes[1].tick_params(top=1, right=1, direction='in')
        axes[2].tick_params(top=1, right=1, direction='in')
        axes[0].legend()
        plt.show()

plt.figure(figsize=(10,5))
plt.plot(learning_curve, color='black', label='learning curve')
plt.plot(test_curve, color='red', label='test curve')
#plt.title("Learning curve")
plt.xlabel('epoch')
plt.ylabel('loss')
plt.tick_params(top=1, right=1, direction='in')
plt.legend()
plt.show()


#ecg_data = create_subseq(ecg[3000:5000], L)
ecg_data = create_subseq(ecg[5000:7000], L)
ecg_data = np.array(ecg_data).astype(np.float32)

_, reconst_seq = enc_dec.forward(ecg_data, train=False)

plt.figure(figsize=(10,5))
plt.plot(np.arange(5000, 7000), ecg_data.ravel(), color='black', label='row data')
plt.plot(np.arange(5000, 7000), reconst_seq.ravel(), color='cyan', label='pred data')
plt.xlabel('time')
plt.ylabel('ECG\'s value')
plt.tick_params(top=1, right=1, direction='in')
plt.legend()
plt.show()



# 異常度の可視化
# computing errors
_, reconst_seq = enc_dec.forward(X_test[:40], train=False)
enc_dec.truncate()
errors = np.abs(X_test[:40] - reconst_seq).flatten()

# estimation
mean = sum(errors)/len(errors)

cov = 0
for e in errors:
    cov += (e - mean)**2
cov /= len(errors)

print('mean : ', mean)
print('cov : ', cov)



mahala_dist = []
for e in errors:
    mahala_dist.append(Mahala_distantce(e, mean, cov))


mahala_dist_sort = sorted(np.array(mahala_dist).ravel(), reverse=True)
th = mahala_dist_sort[59]
print(th)


# マハラノビス距離を計算
def Mahala_distantce(x,mean,cov):
    return (x - mean)**2 / cov
#  異常検知を実行
ecg_data = create_subseq(ecg[3000:5000], L)
ecg_data = np.array(ecg_data).astype(np.float32)
_, reconst_seq = enc_dec.forward(ecg_data, train=False)
errors = np.abs(ecg_data - reconst_seq)
enc_dec.truncate()
mahala_dist = []
for e in errors:
    mahala_dist.append(Mahala_distantce(e, mean, cov))

fig, axes = plt.subplots(nrows=3, figsize=(15,10))
axes[0].plot(np.arange(3000, 5000), ecg_data.ravel(), color='black')
axes[1].plot(np.arange(3000, 5000), reconst_seq.ravel(), color='green')
axes[2].plot(np.arange(3000, 5000), np.array(mahala_dist).ravel(), color='red', label='Mahalanobis Distance')

axes[0].set_ylim(1, 8)

#axes[0].set_title('Original data', fontsize=20)
#axes[1].set_title('Reconstructed data', fontsize=20)
#axes[2].set_title('Anomaly score', fontsize=20)

#axes[0].grid()
#axes[1].grid()
#axes[2].grid()
axes[0].set_ylabel('ECG\'s value')
axes[1].set_ylabel('ECG\'s value')
axes[2].set_ylabel('Mahalanobis Distance')

axes[0].set_xlabel('time')
axes[1].set_xlabel('time')
axes[2].set_xlabel('time')

axes[0].tick_params(top=1, right=1, direction='in')
axes[1].tick_params(top=1, right=1, direction='in')
axes[2].tick_params(top=1, right=1, direction='in')

#x = np.arange(1200, 1400)
#y1 = [1]*len(x)
#y2 = [8]*len(x)
#axes[0].fill_between(x, y1, y2, facecolor='g', alpha=.3)
axes[2].plot([3000,5000], [th,th] , color='black', linestyle='-', label='threshold', linewidth=1)


plt.tight_layout()
plt.legend()
plt.show()
