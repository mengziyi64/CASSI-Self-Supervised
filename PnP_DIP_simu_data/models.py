import torch.nn.functional as F
from torch.autograd import Variable
from my_tools import *
import torch
import torchvision

class UNet_noskip(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet_noskip, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024)
        self.up1 = Up_noskip(1024, 512, bilinear)
        self.up2 = Up_noskip(512, 256, bilinear)
        self.up3 = Up_noskip(256, 128, bilinear)
        self.up4 = Up_noskip(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x = self.inc(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        logits = self.outc(x)
        return logits
    
