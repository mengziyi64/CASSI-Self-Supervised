''' Utilities '''
import math
import numpy as np
import torch
import scipy.io as sio
#import cv2
def TV_denoiser(x, _lambda, n_iter_max):
    dt = 0.25
    N = x.shape
    idx = np.arange(1,N[0]+1)
    idx[-1] = N[0]-1
    iux = np.arange(-1,N[0]-1)
    iux[0] = 0
    ir = np.arange(1,N[1]+1)
    ir[-1] = N[1]-1
    il = np.arange(-1,N[1]-1)
    il[0] = 0
    p1 = np.zeros_like(x)
    p2 = np.zeros_like(x)
    divp = np.zeros_like(x)

    for i in range(n_iter_max):
        z = divp - x*_lambda
        z1 = z[:,ir,:] - z
        z2 = z[idx,:,:] - z
        denom_2d = 1 + dt*np.sqrt(np.sum(z1**2 + z2**2, 2))
        denom_3d = np.tile(denom_2d[:,:,np.newaxis], (1,1,N[2]))
        p1 = (p1+dt*z1)/denom_3d
        p2 = (p2+dt*z2)/denom_3d
        divp = p1-p1[:,il,:] + p2 - p2[iux,:,:]
    u = x - divp/_lambda;
    return u


def A(x, Phi):
    '''
    Forward model of snapshot compressive imaging (SCI), where multiple coded
    frames are collapsed into a snapshot measurement.
    '''
    return np.sum(x*Phi, axis=2)  # element-wise product

def At(y, Phi):
    '''
    Tanspose of the forward model. 
    '''
    # (nrow, ncol, nmask) = Phi.shape
    # x = np.zeros((nrow, ncol, nmask))
    # for nt in range(nmask):
    #     x[:,:,nt] = np.multiply(y, Phi[:,:,nt])
    # return x
    return np.multiply(np.repeat(y[:,:,np.newaxis],Phi.shape[2],axis=2), Phi)

def psnr(ref, img):
    '''
    Peak signal-to-noise ratio (PSNR).
    '''
    mse = np.mean( (ref - img) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 1.
    #PIXEL_MAX = ref.max()
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def psnr_block(ref, img):
    psnr = 0
    r,c,n = img.shape
    PIXEL_MAX = ref.max()
    for i in range(n):
        mse = np.mean( (ref[:,:,i] - img[:,:,i]) ** 2 )
        psnr += 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    return psnr/n

def psnr_torch(ref, img):
    psnr = 0
    for i in range(28):
        mse = torch.mean( (ref[i,:,:] - img[i,:,:]) ** 2 )
        psnr += 20 * torch.log10(1 / mse.sqrt())
    return psnr/28

def shift_back(inputs,step):
    [row,col,nC] = inputs.shape
    for i in range(nC):
        inputs[:,:,i] = np.roll(inputs[:,:,i],(-1)*step*i,axis=1)
    output = inputs[:,0:col-step*(nC-1),:]
    return output

def shift(inputs,step):
    [row,col,nC] = inputs.shape
    output = np.zeros((row, col+(nC-1)*step, nC))
    for i in range(nC):
        output[:,i*step:i*step+col,i] = inputs[:,:,i]
    return output

def A_torch(x,Phi):
    temp = x*Phi
    y = torch.sum(temp,1)
    return y

def At_torch(y,Phi):
    temp = torch.unsqueeze(y, 1).repeat(1,Phi.shape[1],1,1)
    x = temp*Phi
    return x

def shift_torch(inputs,step=1):
    [bs, nC, row, col] = inputs.shape
    output = torch.zeros(bs, nC, row, col+(nC-1)*step)
    for i in range(nC):
        output[:, i, :, i*step:i*step+col] = inputs[:, i,:,:]
    return output.cuda()

def find_max(m,n):
    if m > n:
        return m
    else:
        return n

def gen_log(model_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO) 
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    
    log_file = model_path + '/log.txt'
    fh = logging.FileHandler(log_file, mode='a')
    fh.setLevel(logging.INFO) 
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger  