import torch
import numpy as np
import torch.nn.functional as F

def gen_adj(A):
    D = torch.pow(A.sum(1), -0.5)
    D = torch.diag(D)
    adj = torch.matmul(torch.matmul(A, D).t(), D)
    return adj

def Cos(x, y):
    xy = torch.matmul(x,y.t())
    x_ = (torch.sum(x*x,dim=1) + 1e-9).sqrt()
    y_ = (torch.sum(y*y,dim=1) + 1e-9).sqrt()
    x_ = torch.reshape(x_,[-1,1])
    y_ = torch.reshape(y_,[1,-1])
    xy_ = torch.matmul(x_,y_)
    return xy/xy_
lines = open("key_vector.txt").readlines()

dic = {}
mat = np.zeros([len(lines),200])
count = 0
for iter,line in enumerate(lines[:]):
   tmp = line.strip().split(" ")
   k = tmp[0]
   #if k == "猫" or k == "狗":
   #    print(iter)
   mat[iter] = np.array([float(x) for x in tmp[1:]])
   dic[k] = mat[iter]
mat = torch.from_numpy(mat)
_adj = Cos(mat,mat)
print(_adj)
_adj[_adj<0.4] = 0
_adj[_adj>0.4] = 1
print(_adj)
#_adj = _adj * 0.25 / (_adj.sum(0, keepdims=True))
#_adj = _adj + np.identity(len(lines), np.int)
print(_adj)
_adj = gen_adj(_adj).detach()
print(_adj)
#print(Cos(dic["我"],dic["你"]))
#print(Cos(dic["猫"],dic["狗"]))
