from models import *
from utils import *
import torch.optim as optim
import torch.nn as nn
import torch
import scipy.io as scio
import time
import datetime
import os
import numpy as np
from ssim_torch import ssim
from torch.autograd import Variable

def model_load():
    model = UNet_noskip(24, 24, bilinear=False).cuda()
    #model = FCN_net_updown(28, 28).cuda()
    #model = UNet_ps(28, 28).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001, betas=(0.9, 0.999))
    loss_fn = torch.nn.MSELoss().cuda()
    #loss_fn = torch.nn.L1Loss().cuda()
    return model, optimizer, loss_fn

def get_noise(data_size, noise_type='u', var=1./10):
    shape = [1, data_size[2], data_size[0], data_size[1]]
    net_input = torch.zeros(shape)
    if noise_type == 'u':
        net_input = net_input.uniform_()*var
    elif noise_type == 'n':
        net_input = net_input.normal_()*var
    else:
        assert False
    return net_input.cuda().float()
        
def DIP_denoiser(net_input, ref_truth, Phi_tensor, y_tensor, model, optimizer, loss_fn, iter_num, truth_tensor, Yloss_factor):
    loss_min = torch.tensor([100]).cuda()
    for i in range(iter_num):
        if Yloss_factor == 0:
            model_out = model(net_input)
            optimizer.zero_grad()
            loss = loss_fn(model_out, ref_truth)
            loss.backward()
            optimizer.step()
            if (i+1)%25==0 and loss < loss_min*1.1:
                loss_min = loss
                output = model_out.detach().cpu().numpy()
            if (i+1)%50==0:
                y_hat = A_torch(shift_torch(model_out,2), Phi_tensor)
                Y_mse = loss_fn(y_tensor, y_hat)
                print('DIP iter {}, l2_loss:{:.5f}, y_loss:{:.5f}'.format(i+1, loss.detach().cpu().numpy(), Y_mse.detach().cpu().numpy()))
        else:
            model_out = model(net_input)
            optimizer.zero_grad()
            l2loss = loss_fn(model_out, ref_truth) 
            outshift = shift_torch(model_out, 2)
            y_hat = A_torch(outshift, Phi_tensor)
            yloss = loss_fn(y_hat, y_tensor) * Yloss_factor
            loss = l2loss + yloss
            loss.backward()
            optimizer.step()
            if (i+1)%25==0 and loss < loss_min*1.1:
                loss_min = loss
                output = model_out.detach().cpu().numpy()
            if (i+1)%50==0:
                print('DIP iter {}, l2_loss:{:.5f}, y_loss:{:.5f}'.format(i+1, l2loss.detach().cpu().numpy(), yloss.detach().cpu().numpy()))
    return output
        
