import time
import math
import numpy as np
import matplotlib.pyplot as plt
from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                 denoise_wavelet, estimate_sigma)
from skimage.measure import (compare_psnr, compare_ssim)
from utils import *
from DIP_denoising import *
#import torch
import scipy.io as sio

def ADMM_TV_rec(y, Phi, Phi_sum, maxiter, _lambda, tv_weight, tv_iter_max, eta, shift_step, X_ori):
    begin_time = time.time()
    theta = At(y,Phi)
    v =theta
    b = np.zeros_like(theta)
    for ni in range(maxiter):
        yb = A(theta+b,Phi)
        v  = (theta+b) + np.multiply(_lambda, At( np.divide(y-yb,Phi_sum+eta),Phi))
        temp = shift_back(v-b, shift_step)
        theta = denoise_tv_chambolle(temp, tv_weight, n_iter_max=tv_iter_max, multichannel=True)
        #theta = TV_denoiser(temp, tv_weight, tv_iter_max)
        theta = shift(theta, shift_step)
        b = b-(v-theta)
        weight = 0.999*tv_weight
        eta = 0.998 * eta
        if (ni+1)%5 == 0 and X_ori is None:
            end_time = time.time()
            y_hat = np.sum(theta*Phi,2)
            y_mse = np.mean((y-y_hat)**2)
            print('ADMM-TV, Iteration {}, Y_MSE = {:.5f}dB, time = {}'.format(ni+1, y_mse, (end_time-begin_time)))
        if (ni+1)%5 == 0 and X_ori is not None:
            theta = shift_back(theta, shift_step)
            end_time = time.time()
            print('ADMM-TV: Iteration {}, PSNR = {:.2f}dB, time = {}'.format(ni+1, psnr_block(X_ori, theta), (end_time-begin_time)))
            theta = shift(theta, shift_step)

    return theta





def admm_denoise(y, Phi, Phi_sum, eta=0.01, tao_para = 0.01, denoiser=[], iter_num = 80,
                 tv_weight=0.1, tv_iter_list=[], multichannel=True, 
                 shift_step=2, dip_iter=[], Yloss_factor=0.1, 
                 X_ori=None, index = None, save_path = None):
    Phi_tensor = torch.unsqueeze(torch.from_numpy(np.transpose(Phi, (2, 0, 1))), 0).cuda().float()
    y_tensor = torch.unsqueeze(torch.from_numpy(y), 0).cuda().float()
    if X_ori is not None:
        truth_tensor = torch.from_numpy(np.transpose(X_ori, (2, 0, 1))).cuda().float() 
    else:
        truth_tensor = None
    ymse_min = 1
    p = At(y, Phi) 
    z = At(y, Phi)   # default start point (initialized value)
    x = p
    v = np.zeros_like(p)
    u = np.zeros_like(z)
    net_input = get_noise([660,660,24])
    begin_time = time.time()
    for it in range(iter_num):
        if denoiser[it].lower() == 'tv':
            tao = 0
            z_wave = (eta*(p+v)+tao*(z+u))/(eta+tao)
            yb = A(z_wave, Phi)
            x = z_wave + At(np.divide(y-yb, Phi_sum+eta+tao ),Phi)
            temp = shift_back(x-v, shift_step)
            p = denoise_tv_chambolle(temp, tv_weight, n_iter_max=tv_iter_list[it], multichannel=True)
            x_rec = p
            p = shift(p, shift_step)
            v = v-(x-p)
            tv_weight = 0.999*tv_weight
            eta = 0.99 * eta
            z = p
        if denoiser[it].lower() == 'tv_dip':
            eta = 0
            tao = tao_para
            z_wave = (eta*(p+v)+tao*(z+u))/(eta+tao)
            yb = A(z_wave, Phi)
            x = z_wave + At(np.divide(y-yb, Phi_sum+eta+tao ),Phi)
            if eta != 0:
                ## tv
                temp_p = shift_back(x-v, shift_step)
                p = denoise_tv_chambolle(temp_p, tv_weight, n_iter_max=tv_iter_list[it], multichannel=True)
                p = shift(p, shift_step)
                v = v-(x-p)
                tv_weight = 0.99*tv_weight
                eta = 0.94 * eta
            ## dip
            temp_z = shift_back(x-u, shift_step)
            ref_truth = torch.from_numpy(np.transpose(temp_z, (2, 0, 1)))
            ref_truth = torch.unsqueeze(ref_truth, 0).cuda().float()
            model, optimizer, loss_fn = model_load()
            out = DIP_denoiser(net_input, ref_truth, Phi_tensor, y_tensor, model, optimizer, loss_fn, dip_iter[it], truth_tensor, Yloss_factor)
            z = np.transpose(np.squeeze(out), (1, 2, 0))
            x_rec = z
            z = shift(z, shift_step)
            u = u-(x-z)
            tao = 0.998 * tao
        y_hat = np.sum(z*Phi,2)
        y_mse = np.mean((y-y_hat)**2)
        if (it+1)%5 == 0 or denoiser[it].lower() == 'tv_dip':
            end_time = time.time()
            print('ADMM-{}, Iteration {}, Y_MSE = {:.5f}, time = {}'.format(denoiser[it].upper() ,it+1, y_mse, (end_time-begin_time)))
        if y_mse < ymse_min*1.5 and denoiser[it].lower() == 'tv_dip':
            ymse_min = y_mse
            sio.savemat(save_path + 'scene0{}_{}_{:.3f}.mat'.format(index, it+1, ymse_min),{'pred': x_rec})
    return x_rec