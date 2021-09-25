from models import *
from utils import *
import torch.optim as optim
import torch.nn as nn
import torch
import time
import numpy as np
from torch.autograd import Variable

def model_load():
    model = UNet_noskip(28, 28, bilinear=False).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001, betas=(0.9, 0.999))
    loss_fn = torch.nn.MSELoss().cuda()
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
        
def DIP_denoiser(truth_tensor, net_input, ref_truth, Phi_tensor, y_tensor, model, optimizer, loss_fn, iter_num, mu, rho):
    loss_min = torch.tensor([100]).cuda().float()
    for i in range(iter_num):

        model_out = model(net_input)
        optimizer.zero_grad()
        x_loss = loss_fn(model_out, ref_truth) 
        outshift = shift_torch(model_out, 2)
        y_hat = A_torch(outshift, Phi_tensor)
        y_loss = loss_fn(y_hat, y_tensor) *rho/mu
        loss = x_loss + y_loss
        loss.backward()
        optimizer.step()
        #out = shift_back_torch(outshift, 2)
        if (i+1)%25==0 and y_loss < loss_min*1.1:
            #loss_min = loss
            loss_min = y_loss
            output = model_out.detach().cpu().numpy()
        if (i+1)%100==0:
            PSNR = psnr_torch(truth_tensor, torch.squeeze(model_out))
            print('DIP iter {}, x_loss:{:.5f}, y_loss:{:.5f}, PSNR:{:.2f}'.format(i+1, x_loss.detach().cpu().numpy(), y_loss.detach().cpu().numpy(), PSNR.detach().cpu().numpy()))
            
    return output, loss_min.detach().cpu().numpy()
        
