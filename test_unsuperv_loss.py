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

def just_warp(img, disp):
    B,C,H,W = img.size()
    xx = torch.arange(0, W).view(1,-1).repeat(H,1)
    yy = torch.arange(0,H).view(-1,1).repeat(1,W)
    xx = xx.view(1,1,H,W).repeat(B,1,1,1).float()
    yy = yy.view(1,1,H,W).repeat(B,1,1,1).float()

    xx = xx-disp
    xx = 2.0*xx/max(W-1,1) - 1.0
    yy = 2.0*yy/max(H-1,1) - 1.0
    grid = torch.cat((xx,yy),1).float()

    vgrid = Variable(grid)
    vgrid = vgrid.permute(0,2,3,1)

    output = F.grid_sample(img, vgrid)
    return output

# def just_warp1(img, disp):
#     B,C,H,W = img.size()
#     xx = torch.arange(0, W).view(1,-1).repeat(H,1)
#     yy = torch.arange(0,H).view(-1,1).repeat(1,W)
#     xx = xx.view(1,1,H,W).repeat(B,1,1,1)
#     yy = yy.view(1,1,H,W).repeat(B,1,1,1)
#     grid = torch.cat((xx,yy),1).float()

#     vgrid = Variable(grid)
#     vgrid[:,:1,:,:] = vgrid[:,:1,:,:] - disp

#     vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:]/max(W-1,1) - 1.0
#     vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:]/max(H-1,1) - 1.0
#     vgrid = vgrid.permute(0,2,3,1)

#     output = F.grid_sample(img, vgrid)
#     return output

def get_grid(disp, sh, sw):
    # default sample height & width, and coordinate matrix
    ch = torch.Tensor(range(sh)).unsqueeze(1).repeat(1,sw)
    cw = torch.Tensor(range(sw)).unsqueeze(0).repeat(sh,1)
    coord_matrix = torch.cat((cw.unsqueeze(-1),ch.unsqueeze(-1)),dim=-1)
    mult = torch.ones((sh,sw,2))
    mult[:,:,0] /= ((sw-1)/2)
    mult[:,:,1] /= ((sh-1)/2)

    c = coord_matrix.view(1,sh,sw,2).repeat(disp.size(0),1,1,1)
    c -= torch.cat((disp.unsqueeze(-1),torch.zeros(disp.size(0),sh,sw,1)),dim=-1)
    c = c*mult-1
    return c

model = PSMNet(192)

edgeloss = EdgeAwareLoss()

img_L = imageio.imread('city_training/image_2/012154.png')
img_R = imageio.imread('city_training/image_3/012154.png')

#img_L = imageio.imread('/home/pp456/KITTI/training/image_2/000033_10.png')
#img_R = imageio.imread('/home/pp456/KITTI/training/image_3/000033_10.png')

#img_L,img_R = img_L[64:320,400:912],img_R[64:320,400:912]

imageio.imsave("img_L.png",img_L)
imageio.imsave("img_R.png",img_R)

disp = imageio.imread('city_training/disp/012154.png')
#disp = imageio.imread('/home/pp456/KITTI/training/disp_occ_0/000033_10.png')

disp = np.array(disp,dtype=np.float32)/256.0
#disp = disp[64:320,400:912]

img_L = torch.FloatTensor(img_L).permute(2,0,1).unsqueeze(0)
img_R = torch.FloatTensor(img_R).permute(2,0,1).unsqueeze(0)
disp = torch.FloatTensor(disp).unsqueeze(0)

# # BW
# img_L = torch.mean(img_L,dim=1).unsqueeze(1)
# img_R = torch.mean(img_R,dim=1).unsqueeze(1)

warp = just_warp(img_R,disp)
reverse = just_warp(warp,-disp)
occlude = (reverse+img_R).pow(2) >= 0.01*(reverse.pow(2)+img_R.pow(2))+0.5
disp_mask = (disp>0.0).unsqueeze(1)

disp = disp.unsqueeze(1)
loss_mask = just_warp(torch.ones(img_R.shape),disp)>0.0

loss_mask *= occlude
loss_mask *= disp_mask

loss_l1 = l1_loss(img_L,warp,loss_mask)
loss_edge = edgeloss(img_L,disp,loss_mask)
loss_ssim = ssim_loss(img_L,warp,loss_mask)
# loss_diff = 0.5*(torch.mean((disp[:,:,1:]-disp[:,:,:-1]).pow(2))+torch.mean((disp[:,:,:,1:]-disp[:,:,:,:-1]).pow(2)))

print("loss l1 : " + str(loss_l1))
print("loss edge : " + str(loss_edge))
print("loss ssim : " + str(loss_ssim))

diff = torch.abs(torch.mean(warp,dim=1)-torch.mean(img_L,dim=1))[0]
warped = torch.where(disp_mask.float()>0,warp,img_L)
wrong_warp = torch.where(loss_mask.float()>0,warp,img_R)
imageio.imsave("diff.png",diff.numpy())
imageio.imsave("warped.png",warped[0].permute(1,2,0).numpy())
imageio.imsave("wrong_warp.png",wrong_warp[0].permute(1,2,0).numpy())







# #img_L = imageio.imread('city_training/image_2/012154.png')
# #img_R = imageio.imread('city_training/image_3/012154.png')
# img_L = imageio.imread('/home/pp456/KITTI/training/image_2/000033_10.png')
# img_R = imageio.imread('/home/pp456/KITTI/training/image_3/000033_10.png')
# #img_L = np.pad(img_L,((0,384-img_L.shape[0]),(0,1248-img_L.shape[1]),(0,0)),mode="constant",constant_values=0)
# #img_R = np.pad(img_R,((0,384-img_R.shape[0]),(0,1248-img_R.shape[1]),(0,0)),mode="constant",constant_values=0)
# #ph1 = int((384-img_L.shape[0])/2)
# #ph2 = (384-img_L.shape[0])-ph1
# #pw1 = int((1248-img_L.shape[1])/2)
# #pw2 = (1248-img_L.shape[1])-pw1

# #img_L = np.pad(img_L,((ph1,ph2),(pw1,pw2),(0,0)),mode="constant",constant_values=0)
# #img_R = np.pad(img_R,((ph1,ph2),(pw1,pw2),(0,0)),mode="constant",constant_values=0)

# img_L,img_R = img_R[64:320,400:912],img_L[64:320,400:912]
# #img_L,img_R = img_L[64:320,400:912],img_R[64:320,400:912]

# imageio.imsave("img_L.png",img_L)
# imageio.imsave("img_R.png",img_R)

# disp = imageio.imread('city_training/disp/012154.png')
# #disp = imageio.imread('/home/pp456/KITTI/training/disp_occ_0/000033_10.png')
# disp = np.array(disp,dtype=np.float32)/256.0
# #disp = np.load('utils/depth.npy')
# oh,ow = disp.shape[0],disp.shape[1]
# #disp = np.pad(disp,((0,384-disp.shape[0]),(0,1248-disp.shape[1])),mode="constant",constant_values=0)
# #disp = np.where(disp==0,0.0,0.54*721/disp)

# disp = disp[64:320,400:912]

# img_L = torch.FloatTensor(img_L).permute(2,0,1).unsqueeze(0)
# img_R = torch.FloatTensor(img_R).permute(2,0,1).unsqueeze(0)
# disp = torch.FloatTensor(disp).unsqueeze(0)

# # warp full image

# # coord = get_grid(disp, 256, 512)
# warp = just_warp(img_L,disp)
# # warp = F.grid_sample(img_L,coord,mode="bilinear",padding_mode="border")
# reverse = just_warp(warp,-disp)
# #reverse = F.grid_sample(warp,get_grid(-disp, 256, 512),mode="bilinear",padding_mode="border")
# occlude = (reverse+img_L).pow(2) >= 0.01*(reverse.pow(2)+img_L.pow(2))+0.5
# disp_mask = (disp>0.0).unsqueeze(1)

# disp = disp.unsqueeze(1)
# loss_mask = just_warp(torch.ones(img_R.shape),disp)>0.0
# #loss_mask = F.grid_sample(torch.ones(img_R.shape),coord.cpu(),padding_mode="zeros")>0.0
# loss_mask *= occlude
# print("loss mask % without disp : " + str(torch.mean(loss_mask.float())))
# loss_mask *= disp_mask
# print("loss mask % with disp : " + str(torch.mean(loss_mask.float())))
# print("disp mask % : " + str(torch.mean(disp_mask.float())))

# loss_l1 = l1_loss(img_R,warp,loss_mask)
# loss_edge = edgeloss(img_L,disp,loss_mask)
# loss_ssim = ssim_loss(img_R,warp,loss_mask)
# loss_diff = 0.5*(torch.mean((disp[:,:,1:]-disp[:,:,:-1]).pow(2))+torch.mean((disp[:,:,:,1:]-disp[:,:,:,:-1]).pow(2)))

# print("loss l1 : " + str(loss_l1))
# print("loss edge : " + str(loss_edge))
# print("loss ssim : " + str(loss_ssim))

# diff = torch.abs(torch.mean(warp,dim=1)-torch.mean(img_R,dim=1))[0]
# warped = torch.where(disp_mask.float()>0,warp,img_R)
# wrong_warp = torch.where(loss_mask.float()>0,warp,img_L)
# imageio.imsave("diff.png",diff.numpy())
# imageio.imsave("warped.png",warped[0].permute(1,2,0).numpy())
# imageio.imsave("wrong_warp.png",wrong_warp[0].permute(1,2,0).numpy())
