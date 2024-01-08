import os, sys
import time, math
import argparse, random
from math import exp
import numpy as np

import torch
from torch import nn, optim
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.data import DataLoader
from torch.backends import cudnn
from torch.autograd import Variable

import torchvision
import torchvision.transforms as tfs
from torchvision.transforms import ToPILImage
from torchvision.transforms import functional as FF
import torchvision.utils as vutils
from torchvision.utils import make_grid
from torchvision.models import vgg16

from PIL import Image
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class Metrics:

    def __init__(self):
        pass
    
    def gaussian(self,window_size, sigma):
        gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()

    def create_window(self,window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
        del _1D_window, _2D_window
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return window

    def _ssim(self,img1, img2, window, window_size, channel, size_average=True):
        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        if torch.cuda.is_available():
            mu1.detach().cpu(),mu2.detach().cpu(), mu1_sq.detach().cpu(), mu2_sq.detach().cpu(), mu1_mu2.detach().cpu(), sigma1_sq.detach().cpu(),sigma2_sq.detach().cpu(), sigma12.detach().cpu()

        del mu1,mu2, mu1_sq, mu2_sq, mu1_mu2, sigma1_sq,sigma2_sq, sigma12

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

    def ssim(self,img1, img2, window_size=11, size_average=True):
        img1=torch.clamp(img1,min=0,max=1)
        img2=torch.clamp(img2,min=0,max=1)
        (_, channel, _, _) = img1.size()
        window = self.create_window(window_size, channel)
        if img1.is_cuda:
            window = window.cuda(img1.get_device())
        window = window.type_as(img1)
        ans = self._ssim(img1, img2, window, window_size, channel, size_average)

        if torch.cuda.is_available():
            img1.detach().cpu(), img2.detach().cpu()
        
        del img1, img2, window, window_size, channel, size_average

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return ans

    def psnr(self,pred, gt):
        pred=pred.clamp(0,1).detach().cpu().numpy()
        gt=gt.clamp(0,1).detach().cpu().numpy()
        imdff = pred - gt
        rmse = math.sqrt(np.mean(imdff ** 2))
        
        del pred, gt, imdff
        torch.cuda.empty_cache()
        if rmse == 0:
            return 100
        return 20 * math.log10( 1.0 / rmse)