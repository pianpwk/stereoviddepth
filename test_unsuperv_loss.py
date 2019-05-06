import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time
import math
import argparse
import os
import random
from lib.model import DRNSegment,PSMNet
from utils.dataloader import *
from PIL import Image
from loss import l1_loss,ssim_loss,EdgeAwareLoss
from eval_utils import end_point_error
import sys
import imageio
sys.path.append('drnseg')
sys.path.append('lib')

# cuda
use_cuda = torch.cuda.is_available()

def get_grid(disp, sh, sw):
    # default sample height & width, and coordinate matrix
    ch = torch.Tensor(range(sh)).unsqueeze(1).repeat(1,sw)
    cw = torch.Tensor(range(sw)).unsqueeze(0).repeat(sh,1)
    coord_matrix = torch.cat((cw.unsqueeze(-1),ch.unsqueeze(-1)),dim=-1)
    mult = torch.ones((sh,sw,2))
    mult[:,:,0] /= ((sw-1)/2)
    mult[:,:,1] /= ((sh-1)/2)
    if use_cuda:
        coord_matrix,mult = coord_matrix.cuda(),mult.cuda()

    c = coord_matrix.view(1,sh,sw,2).repeat(disp.size(0),1,1,1)
    c -= torch.cat((disp.unsqueeze(-1),torch.zeros(disp.size(0),sh,sw,1).cuda()),dim=-1)
    c = c*mult-1
    return c

model = PSMNet(192)

if use_cuda:
    model = nn.DataParallel(model)
    model.cuda()

edgeloss = EdgeAwareLoss()
if use_cuda:
    edgeloss = edgeloss.cuda()

#img_L = imageio.imread('city_training/image_2/012154.png')
#img_R = imageio.imread('city_training/image_3/012154.png')
img_L = imageio.imread('/home/pp456/KITTI/training/image_2/000033_10.png')
img_R = imageio.imread('/home/pp456/KITTI/training/image_3/000033_10.png')
#img_L = np.pad(img_L,((0,384-img_L.shape[0]),(0,1248-img_L.shape[1]),(0,0)),mode="constant",constant_values=0)
#img_R = np.pad(img_R,((0,384-img_R.shape[0]),(0,1248-img_R.shape[1]),(0,0)),mode="constant",constant_values=0)
#ph1 = int((384-img_L.shape[0])/2)
#ph2 = (384-img_L.shape[0])-ph1
#pw1 = int((1248-img_L.shape[1])/2)
#pw2 = (1248-img_L.shape[1])-pw1

#img_L = np.pad(img_L,((ph1,ph2),(pw1,pw2),(0,0)),mode="constant",constant_values=0)
#img_R = np.pad(img_R,((ph1,ph2),(pw1,pw2),(0,0)),mode="constant",constant_values=0)

img_L,img_R = img_R[64:320,400:912],img_L[64:320,400:912]
#img_L,img_R = img_L[64:320,400:912],img_R[64:320,400:912]

imageio.imsave("img_L.png",img_L)
imageio.imsave("img_R.png",img_R)

disp = imageio.imread('city_training/disp/012154.png')
#disp = imageio.imread('/home/pp456/KITTI/training/disp_occ_0/000033_10.png')
disp = np.array(disp,dtype=np.float32)/256.0
#disp = np.load('utils/depth.npy')
oh,ow = disp.shape[0],disp.shape[1]
#disp = np.pad(disp,((0,384-disp.shape[0]),(0,1248-disp.shape[1])),mode="constant",constant_values=0)
#disp = np.where(disp==0,0.0,0.54*721/disp)

disp = disp[64:320,400:912]

img_L = torch.FloatTensor(img_L).permute(2,0,1).unsqueeze(0).cuda()
img_R = torch.FloatTensor(img_R).permute(2,0,1).unsqueeze(0).cuda()
disp = torch.FloatTensor(disp).unsqueeze(0).cuda()

# warp full image

coord = get_grid(disp, 256, 512)
warp = F.grid_sample(img_L,coord,mode="bilinear",padding_mode="border")
reverse = F.grid_sample(warp,get_grid(-disp, 256, 512),mode="bilinear",padding_mode="border")
occlude = (reverse+img_L).pow(2) >= 0.01*(reverse.pow(2)+img_L.pow(2))+0.5
disp_mask = (disp>0.0).unsqueeze(1)

disp = disp.unsqueeze(1)
loss_mask = F.grid_sample(torch.ones(img_R.shape),coord.cpu(),padding_mode="zeros")>0.0
loss_mask *= occlude.cpu()
print("loss mask % without disp : " + str(torch.mean(loss_mask.float())))
loss_mask *= disp_mask.cpu()
print("loss mask % with disp : " + str(torch.mean(loss_mask.float())))
loss_mask = loss_mask.cuda()
print("disp mask % : " + str(torch.mean(disp_mask.float())))

loss_l1 = l1_loss(img_R,warp,loss_mask)
loss_edge = edgeloss(img_L,disp,loss_mask)
loss_ssim = ssim_loss(img_R,warp,loss_mask)
loss_diff = 0.5*(torch.mean((disp[:,:,1:]-disp[:,:,:-1]).pow(2))+torch.mean((disp[:,:,:,1:]-disp[:,:,:,:-1]).pow(2)))

print("loss l1 : " + str(loss_l1))
print("loss edge : " + str(loss_edge))
print("loss ssim : " + str(loss_ssim))

diff = torch.abs(torch.mean(warp,dim=1)-torch.mean(img_R,dim=1))[0]
warped = torch.where(disp_mask.cpu().float().cuda()>0,warp,img_R)
wrong_warp = torch.where(loss_mask.float().cuda()>0,warp,img_L)
imageio.imsave("diff.png",diff.cpu().numpy())
imageio.imsave("warped.png",warped[0].permute(1,2,0).cpu().numpy())
imageio.imsave("wrong_warp.png",wrong_warp[0].permute(1,2,0).cpu().numpy())
