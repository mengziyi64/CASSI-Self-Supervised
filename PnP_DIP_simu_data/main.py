import os
import time
import math
import numpy as np
from numpy import *
import scipy.io as sio
from statistics import mean
from PnP_DIP import *
from utils import *
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

sample = '01'
maskfile = '../Data/mask/mask_3d_shift.mat'
save_path = './Result/result'+ sample + '/'
if not os.path.exists(save_path):
    os.makedirs(save_path)
r, c, nC = 256, 256, 28
Phi = sio.loadmat(maskfile)['mask_3d_shift']
Phi_sum = np.sum(Phi**2,2)
Phi_sum[Phi_sum==0]=1
index = int(sample)
datapath = '../Data/kaist_data/scene'+ sample + '.mat'
X_ori = sio.loadmat(datapath)['img']
X_ori = X_ori/X_ori.max()
X_ori_shift = shift(X_ori, step=2)
y = A(X_ori_shift,Phi)
tvdip_num =  60
mu = 0.01
eta = 0
denoiser = 'DIP'
iter_num = tvdip_num
tv_weight = 0.1
tv_iter_num = 5
dip_iter = [500]*10 + [700]*20 +[1200]*30
shift_step = 2
rho = 0.001
x_rec = admm_dip(y, Phi, Phi_sum, eta=eta, mu=mu, rho=rho,
                     denoiser=denoiser, iter_num=iter_num, 
                     tv_weight=tv_weight, tv_iter_num=tv_iter_num,
                     multichannel=True, shift_step=shift_step, 
                     dip_iter=dip_iter, index = index, X_ori=X_ori, save_path = save_path)
