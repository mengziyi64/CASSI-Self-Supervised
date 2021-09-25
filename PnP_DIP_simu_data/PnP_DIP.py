import time
import math
import numpy as np
from skimage.restoration import denoise_tv_chambolle
from utils import *
from DIP_denoising import *
import scipy.io as sio



def admm_dip(y, Phi, Phi_sum, eta=0.01, mu = 0.01, rho = 0.001, denoiser=[], iter_num = 80,
                 tv_weight=0.1, tv_iter_num=5, multichannel=True, 
                 shift_step=2, dip_iter=[], index = None, X_ori=None, save_path = None):
    
    Phi_tensor = torch.unsqueeze(torch.from_numpy(np.transpose(Phi, (2, 0, 1))), 0).cuda().float()
    y_tensor = torch.unsqueeze(torch.from_numpy(y), 0).cuda().float() 
    truth_tensor = torch.from_numpy(np.transpose(X_ori, (2, 0, 1))).cuda().float() 
    psnr_max = 0
    u = At(y, Phi) 
    T = At(y, Phi)   # default start point (initialized value)
    x = u
    v = np.zeros_like(u)
    b = np.zeros_like(T)
    net_input = get_noise(X_ori.shape)
    begin_time = time.time()
    loss_y_min = 1
    for it in range(iter_num):
        c = (eta*(u+v)+mu*(T+b))/(eta+mu)
        yb = A(c, Phi)
        x = c + At(np.divide(y-yb, Phi_sum+eta+mu ),Phi)
        if eta != 0:
            denoiser = 'DIP-TV'
            ## tv
            temp_u = shift_back(x-v, shift_step)
            u = denoise_tv_chambolle(temp_u, tv_weight, n_iter_max=tv_iter_num, multichannel=True)
            u = shift(u, shift_step)
            v = v-(x-u)
            tv_weight = 0.99*tv_weight
            eta = 0.92 * eta
        ## dip
        temp_T = shift_back(x-b, shift_step)
        ref_truth = torch.from_numpy(np.transpose(temp_T, (2, 0, 1)))
        ref_truth = torch.unsqueeze(ref_truth, 0).cuda().float()
        model, optimizer, loss_fn = model_load()
        out, loss_y_iter = DIP_denoiser(truth_tensor, net_input, ref_truth, Phi_tensor, y_tensor, model, optimizer, loss_fn, dip_iter[it], mu, rho)
        T = np.transpose(np.squeeze(out), (1, 2, 0))
        x_rec = T
        T = shift(T, shift_step)
        b = b-(x-T)
        mu = 0.998 * mu
        psnr_x = psnr_block(X_ori, x_rec)
        
        end_time = time.time()
        print('PnP-{}, Iteration {}, loss = {:.5f}, PSNR = {:.2f}dB, time = {}'.format(denoiser ,it+1, loss_y_iter, psnr_x, (end_time-begin_time)))
        if loss_y_iter < loss_y_min and it > 40:
            loss_y_min = loss_y_iter
            sio.savemat(save_path + 'scene0{}_{}_{:.2f}.mat'.format(index, it+1, psnr_x),{'x_rec': x_rec})
    return x_rec



