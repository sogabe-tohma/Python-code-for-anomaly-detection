import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import time
np.random.seed(111)

if __name__ == '__main__':
   start = time.time()

n = 40
x,y=np.zeros((3,n)),np.zeros(n)
x[0,0:20]=np.random.randn(n//2)-15
x[0,20::]=np.random.randn(n//2)-5
x[1,:]=(np.random.randn(n))
y[0:20]=np.ones(n//2)
y[20::]=-np.ones(n//2)
x[0,0:2]=x[0,0:2]+50
x[2,::]=1
l,e=0.0001,0.001
t0=np.zeros((3,1))
for i in range(1000):
    m=np.dot(t0.T,x)*y
    v=m+np.min([np.ones_like(m),np.max([np.zeros_like(m),1-m],axis=0)],axis=0)
    a=np.abs(v-m)
    w=np.ones_like(y)
    for j in range(len(w)):
        if a[0,j]>e:
            w[j]=e/a[0,j]
    w=np.reshape(w,[40,1])
    t1= LA.inv(np.dot(x,(np.repeat(w,3,axis=1).T*x).T))+l*np.eye(3)
    t2=np.dot(x,(w.T*v*y).T)
    t=np.dot(t1,t2)
    if LA.norm(t-t0)<0.1:
        break
    t0=t
print(t)
x1_min = np.amin(x[1,0:20])
x1_max = np.amax(x[1,20::])
z=[-16,5]
plt.scatter(x[0,0:20], x[1,0:20], marker='o',c=y[0:20])
plt.scatter(x[0,20::], x[1,20::], marker='x',c=y[20::])
plt.plot(z,-(t[2]+z*t[0])/t[1],'k-')
plt.ylim([x1_min-2,x1_max+2])
plt.show()
