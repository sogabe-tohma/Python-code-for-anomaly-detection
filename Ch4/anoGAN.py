import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

from mnist import MNIST

import renom as rm
from renom.optimizer import Adam
from renom.cuda import set_cuda_active
set_cuda_active(False)



# データセット読み込み
mnist = MNIST('./MNIST')
x_train, y_train = mnist.load_training()
x_test, y_test = mnist.load_testing()

fashion = MNIST('./Fashion_MNIST')
x_fashion, y_fashion = fashion.load_testing()



# データの前処理
x_train = np.asarray(x_train).astype(np.float32)
y_train = np.asarray(y_train).astype(np.int32)
x_test = np.asarray(x_test).astype(np.float32)
y_test = np.asarray(y_test).astype(np.int32)
x_fashion=np.asarray(x_fashion).astype(np.float32)
y_fashion=np.asarray(y_fashion).astype(np.int32)

#  画像データを0〜1にリスケール
x_train = x_train.reshape(len(x_train), 1, 28, 28) / 255.0
x_test = x_test.reshape(len(x_test), 1, 28, 28) / 255.0
x_fashion = x_fashion.reshape(len(x_fashion), 1, 28, 28) / 255.0



def imshow(image_set, nrows=4, ncols=10, figsize=(12.5, 5), save=False):
    plot_num = nrows * ncols
    fig, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=(12, 12*nrows/ncols))
    plt.tight_layout(False)
    fig.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
    ax = ax.ravel()
    for i in range(plot_num):
        ax[i].imshow(-image_set[i].reshape(28, 28), cmap='gray')
        ax[i].set_xticks([])
        ax[i].set_yticks([])
    if save is not False:
        plt.savefig(save + ".png")

imshow(x_train)
plt.show()
imshow(x_fashion)
plt.show()



# ハイパーパラメータの設定
batch_size = 128
lam = 0.1
gan_epoch = 200
Gamma = 200
dim_z = 100          # of the dimension of latent variable
slope = 0.02          # slope of leaky relu
b1 = 0.5                 # momentum term of adam
lr1 = 0.001            # initial learning rate for adam
lr2 = 0.0001         # subsequent learning rate for adam



# 学習モデル定義
class discriminator(rm.Model):

    def __init__(self):
        super(discriminator, self).__init__()
        self._conv1 = rm.Conv2d(channel=8, filter=3, padding=1, stride=1)
        self._conv2 = rm.Conv2d(channel=16, filter=2, padding=1, stride=2)
        self._conv3 = rm.Conv2d(channel=32, filter=3, padding=1, stride=2)
        self._conv4 = rm.Conv2d(channel=64, filter=2, padding=1, stride=2)
        self._conv5 = rm.Conv2d(channel=128, filter=3, padding=1, stride=2)
        self.dr1 = rm.Dropout()
        self.dr2 = rm.Dropout()
        self.dr3 = rm.Dropout()
        self.dr4 = rm.Dropout()
        self.dr5 = rm.Dropout()
        self.fl = rm.Flatten()
        self._full = rm.Dense(output_size=1)

    def forward(self, x):
        self.fx = rm.leaky_relu(self._conv1(x), slope=slope)
        h1  = self.dr1(self.fx)
        h2 = self.dr2(rm.leaky_relu(self._conv2(h1), slope=slope))
        h3 = self.dr3(rm.leaky_relu(self._conv3(h2), slope=slope))
        h4 = self.dr4(rm.leaky_relu(self._conv4(h3), slope=slope))
        h5 = self.dr5(rm.leaky_relu(self._conv5(h4), slope=slope))
        h6 = self.fl(h5)
        h7 = self._full(h6)
        return h7

class generator(rm.Model):

    def __init__(self):
        super(generator, self).__init__()
        self._deconv2 = rm.Deconv2d(channel=64, filter=2, stride=2)
        self._deconv3 = rm.Deconv2d(channel=32, filter=3, stride=2)
        self._deconv4 = rm.Deconv2d(channel=16, filter=4, stride=2)
        self._deconv5 = rm.Deconv2d(channel=1, filter=1, stride=1)
        self._full = rm.Dense(input_size=dim_z, output_size=128*3*3)
        self.bn2 = rm.BatchNormalize()
        self.bn3 = rm.BatchNormalize()
        self.bn4 = rm.BatchNormalize()

    def forward(self, z):
        z = rm.reshape(self._full(z), (-1, 128, 3, 3))
        h1 = rm.relu(self._deconv2(self.bn2(z)))
        h2 = rm.relu(self._deconv3(self.bn3(h1)))
        h3 = rm.relu(self._deconv4(self.bn4(h2)))
        h4 = self._deconv5(h3)
        h5 = rm.tanh(h4)
        return h5

class dcgan(rm.Model):
    def __init__(self, gen, dis, minimax=False):
        self.gen = gen
        self.dis = dis
        self.minimax = minimax

    def forward(self, x):
        batch = len(x)
        z = np.random.randn(batch*dim_z).reshape((batch, dim_z)).astype(np.float32)
        self.x_gen = self.gen(z)
        self.real_dis = self.dis(x)
        self.fake_dis = self.dis(self.x_gen)
        self.prob_real = rm.sigmoid(self.real_dis)
        self.prob_fake = rm.sigmoid(self.fake_dis)
        self.dis_loss_real = rm.sigmoid_cross_entropy(self.real_dis, np.ones(batch).reshape(-1,1))
        self.dis_loss_fake = rm.sigmoid_cross_entropy(self.fake_dis, np.zeros(batch).reshape(-1,1))
        self.dis_loss = self.dis_loss_real + self.dis_loss_fake
        if self.minimax:
            self.gen_loss = -self.dis_loss
        else: #non-saturating
            self.gen_loss = rm.sigmoid_cross_entropy(self.fake_dis, np.ones(batch).reshape(-1,1))

        return self.dis_loss



# 訓練
from tqdm import trange

dis_opt = rm.Adam(b=b1, lr=lr1)
gen_opt = rm.Adam(b=b1, lr=lr1)

dis = discriminator()
gen = generator()
gan = dcgan(gen, dis)

N = len(x_train)

loss_curve_dis = []
loss_curve_gen = []
acc_curve_real = []
acc_curve_fake = []

for epoch in trange(1, gan_epoch+1):
    perm = np.random.permutation(N)
    total_loss_dis = 0
    total_loss_gen = 0
    total_acc_real = 0
    total_acc_fake = 0

    if epoch <= (gan_epoch - (gan_epoch//2)):
        dis_opt._lr = lr1 - (lr1 - lr2) * epoch / (gan_epoch - (gan_epoch // 2))
        gen_opt._lr = lr1 - (lr1 - lr2) * epoch / (gan_epoch - (gan_epoch // 2))

    for i in range(N // batch_size):
        index = perm[i*batch_size : (i+1)*batch_size]
        train_batch = x_train[index]
        with gan.train():
            dl = gan(train_batch)
        with gan.gen.prevent_update():
            dl = gan.dis_loss
            dl.grad(detach_graph=False).update(dis_opt)
        with gan.dis.prevent_update():
            gl = gan.gen_loss
            gl.grad().update(gen_opt)
        real_acc = len(np.where(gan.prob_real.as_ndarray() >= 0.5)[0]) / batch_size
        fake_acc = len(np.where(gan.prob_fake.as_ndarray() < 0.5)[0]) / batch_size
        dis_loss_ = gan.dis_loss.as_ndarray()#[0]
        gen_loss_ = gan.gen_loss.as_ndarray()#[0]
        total_loss_dis += dis_loss_
        total_loss_gen += gen_loss_
        total_acc_real += real_acc
        total_acc_fake += fake_acc
    loss_curve_dis.append(total_loss_dis/(N//batch_size))
    loss_curve_gen.append(total_loss_gen/(N//batch_size))
    acc_curve_real.append(total_acc_real/(N//batch_size))
    acc_curve_fake.append(total_acc_fake/(N//batch_size))


    if epoch%10 == 0:
        print("Epoch {}  Loss of Dis {:.3f} Loss of Gen {:.3f} Accuracy of Real {:.3f} Accuracy of Fake {:.3f}".format(
                 epoch, loss_curve_dis[-1], loss_curve_gen[-1], acc_curve_real[-1], acc_curve_fake[-1]))



# 結果プロット
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
ax = ax.ravel()
ax[0].plot(loss_curve_dis, linewidth=2, label="dis")
ax[0].plot(loss_curve_gen, linewidth=2, label="gen")
ax[0].set_title("Learning curve")
ax[0].set_ylabel("loss")
ax[0].set_xlabel("epoch")
ax[0].legend()
ax[0].grid()
ax[1].plot(acc_curve_real, linewidth=2, label="real")
ax[1].plot(acc_curve_fake, linewidth=2, label="fake")
ax[1].set_title("Accuracy curve")
ax[1].set_ylabel("accuracy")
ax[1].set_xlabel("epoch")
ax[1].legend()
ax[1].grid()

ncols = 10
nrows = 4
z = np.random.randn(ncols*nrows*dim_z).reshape((ncols*nrows, dim_z)).astype(np.float32)
gen_images = gan.gen(z).as_ndarray()
imshow(gen_images)
plt.show()



# 潜在変数の探索
def res_loss(x, z):
    Gz = gan.gen(z)
    abs_sub = abs(x - Gz)
    return rm.sum(abs_sub)


def dis_loss(x, z):
    # compute f(x)
    dl = gan.dis(x)
    fx = gan.dis.fx.as_ndarray()

    # compute f(G(z))
    Gz = gan.gen(z)
    dl = gan.dis(Gz)
    G_fx = gan.dis.fx

    abs_sub = abs(fx - G_fx)
    return rm.sum(abs_sub)


def Loss(x, z):
    return (1-lam)*res_loss(x, z) + lam*dis_loss(x, z)

def numerical_diff(f, x, z):
    with gan.train():
        loss = f(x,z)
        diff = loss.grad().get(z)
    return np.array(diff)
def grad_descent(f, x, niter=Gamma):
    z_gamma = rm.Variable(np.random.randn(dim_z).reshape((1, dim_z)).astype(np.float32))
    lr = 0.1
    for _ in range(niter):
        z_gamma -= lr*numerical_diff(Loss, x, z_gamma)
    return z_gamma


def Anomaly_score(image_set, nrows=1, ncols=5, figsize=(12.5, 5), normal=True):
    plot_num = nrows * ncols
    _, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=(12, 12*nrows/ncols))
    plt.tight_layout()
    ax = ax.ravel()
    for i in range(plot_num):
        idx = np.random.choice(np.arange(image_set.shape[0]))
        ax[i].imshow(-image_set[idx].reshape(28, 28), cmap='gray')
        x = image_set[idx].reshape((1,1,28,28))
        z_Gamma = grad_descent(Loss, x)
        a_score = Loss(x, z_Gamma)
        if normal:
            #print(a_score)
            ax[i].set_title("Normal: \n"+str(a_score))
        else:
            ax[i].set_title("Anomalous: \n"+str(a_score))
        ax[i].set_xticks([])
        ax[i].set_yticks([])


gan.set_models(inference=True)
Anomaly_score(x_test)
Anomaly_score(x_fashion, normal=False)



scores = []
plot_num = 1000
for i in range(plot_num):
    idx = np.random.choice(np.arange(x_test.shape[0]))
    #ax[i].imshow(-image_set[idx].reshape(28, 28), cmap='gray')
    x = x_test[idx].reshape((1,1,28,28))
    z_Gamma = grad_descent(Loss, x)
    scores.append(Loss(x, z_Gamma))

# print(scores)

scores_sort = sorted(scores, reverse=True)
th = scores_sort[29]
print(th)
#print(scores[0])
