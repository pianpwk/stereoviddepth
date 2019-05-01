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
from utils.dataloader import StereoSeqDataset,StereoSupervDataset
from loss import *
from eval_utils import end_point_error
import sys
import imageio
import matplotlib.pyplot as plt
from PIL import Image
sys.path.append('drnseg')
sys.path.append('lib')

# cuda
use_cuda = torch.cuda.is_available()

# default sample height & width, and coordinate matrix
sh,sw = 256,512
ch = torch.Tensor(range(sh)).unsqueeze(1).repeat(1,sw)
cw = torch.Tensor(range(sw)).unsqueeze(0).repeat(sh,1)
coord_matrix = torch.cat((cw.unsqueeze(-1),ch.unsqueeze(-1)),dim=-1)
mult = torch.ones((sh,sw,2))
mult[:,:,0] /= (sw-1)/2
mult[:,:,1] /= (sh-1)/2

if use_cuda:
    coord_matrix,mult = coord_matrix.cuda(),mult.cuda()

def get_grid(disp):
    c = coord_matrix.view(1,sh,sw,2).repeat(disp.size(0),1,1,1)
    c += torch.cat((disp.unsqueeze(-1),torch.zeros(disp.size(0),sh,sw,1).cuda()),dim=-1)
    c = c*mult-1
    return c
    
model = PSMNet(192)
model = nn.DataParallel(model).cuda()

optimizer = optim.Adam(model.parameters(), lr=0.001)

import utils.psmprocess as psmprocess

preprocess = psmprocess.get_transform(augment=False)
ch,cw = 256,512
def get_image(f,x1,y1):
    img = Image.open(f).convert('RGB')
    img = preprocess(img.crop((x1,y1,x1+cw,y1+ch)))
    return img
    
def get_disp(f,x1,y1):
    disp = np.array(imageio.imread(f),dtype=np.float32)/256.0
    disp = disp[y1:y1+ch,x1:x1+cw]
    disp = torch.FloatTensor(disp)
    return disp
    
# data

x1 = random.randint(0, 1230 - cw)
y1 = random.randint(0, 360 - ch)
x1,y1 = 300,20
imgL_0 = get_image('some_kitti/0_2.png',x1,y1)
imgR_0 = get_image('some_kitti/0_3.png',x1,y1)
disp_0 = get_disp('some_kitti/0_depth.png',x1,y1)

x1 = random.randint(0, 1230 - cw)
y1 = random.randint(0, 360 - ch)
imgL_1 = get_image('some_kitti/3_2.png',x1,y1)
imgR_1 = get_image('some_kitti/3_3.png',x1,y1)
disp_1 = get_disp('some_kitti/3_depth.png',x1,y1)

x1 = random.randint(0, 1230 - cw)
y1 = random.randint(0, 360 - ch)
imgL_2 = get_image('some_kitti/7_2.png',x1,y1)
imgR_2 = get_image('some_kitti/7_3.png',x1,y1)
disp_2 = get_disp('some_kitti/7_depth.png',x1,y1)

x1 = random.randint(0, 1230 - cw)
y1 = random.randint(0, 360 - ch)
imgL_3 = get_image('some_kitti/9_2.png',x1,y1)
imgR_3 = get_image('some_kitti/9_3.png',x1,y1)
disp_3 = get_disp('some_kitti/9_depth.png',x1,y1)

x1 = random.randint(0, 1230 - cw)
y1 = random.randint(0, 360 - ch)
imgL_4 = get_image('some_kitti/10_2.png',x1,y1)
imgR_4 = get_image('some_kitti/10_3.png',x1,y1)
disp_4 = get_disp('some_kitti/10_depth.png',x1,y1)

imageio.imsave("train_loss_outputs/imgL.png",imgL_0.permute(1,2,0).numpy())
imageio.imsave("train_loss_outputs/imgR.png",imgR_0.permute(1,2,0).numpy())

edgeloss = EdgeAwareLoss().cuda()

# train with 1 image

for i in range(1000):
    
    optimizer.zero_grad()
    
    xL = imgL_0.unsqueeze(0).cuda()
    xR = imgR_0.unsqueeze(0).cuda()
    y = disp_0.unsqueeze(0).cuda()

    y = y.squeeze(1)
    mask = (y < 192)*(y > 0.0)
    mask.detach_()
    
    output1, output2, output3 = model(xL,xR) # L-R input
    output1 = torch.squeeze(output1,1)
    output2 = torch.squeeze(output2,1)
    output3 = torch.squeeze(output3,1)

    s_loss = 0.5*F.smooth_l1_loss(output1[mask], y[mask], size_average=True) + 0.7*F.smooth_l1_loss(output2[mask], y[mask], size_average=True) + F.smooth_l1_loss(output3[mask], y[mask], size_average=True)
    #s_loss.backward()

    coord1 = get_grid(output1)
    coord2 = get_grid(output2)
    coord3 = get_grid(output3)

    xL,xR = torch.mean(xL,dim=1).unsqueeze(1),torch.mean(xR,dim=1).unsqueeze(1)    

    warp1 = F.grid_sample(xL,coord1,mode="bilinear",padding_mode="border")
    warp2 = F.grid_sample(xL,coord2,mode="bilinear",padding_mode="border")
    warp3 = F.grid_sample(xL,coord3,mode="bilinear",padding_mode="border")

    output1,output2,output3 = output1.unsqueeze(1),output2.unsqueeze(1),output3.unsqueeze(1)

    loss1_mask = F.grid_sample(torch.ones(xR.shape).cuda(),coord1,padding_mode="zeros")>0.0
    loss2_mask = F.grid_sample(torch.ones(xR.shape).cuda(),coord2,padding_mode="zeros")>0.0
    loss3_mask = F.grid_sample(torch.ones(xR.shape).cuda(),coord3,padding_mode="zeros")>0.0

    #u_loss = (0.5*l1_loss(xR,warp1,loss1_mask)+0.7*l1_loss(xR,warp2,loss2_mask)+l1_loss(xR,warp3,loss3_mask))
    #u_loss.backward()
    #output3 = output3.squeeze(1)

    u_loss = 0.5*(l1_loss(xR,warp1,loss1_mask)+0.5*edgeloss(xL,output1,loss1_mask)+0.5*ssim_loss(xR,warp1,loss1_mask))
    u_loss += 0.7*(l1_loss(xR,warp2,loss2_mask)+0.5*edgeloss(xL,output2,loss2_mask)+0.5*ssim_loss(xR,warp2,loss2_mask))
    u_loss += l1_loss(xR,warp3,loss3_mask)+0.5*edgeloss(xL,output3,loss3_mask)+0.5*ssim_loss(xR,warp3,loss3_mask)

    diff_loss = 0.5*(torch.mean((output1[:,:,1:]-output1[:,:,:-1]).pow(2))+torch.mean((output1[:,:,:,1:]-output1[:,:,:,:-1]).pow(2)))
    diff_loss += 0.7*(torch.mean((output2[:,:,1:]-output2[:,:,:-1]).pow(2))+torch.mean((output2[:,:,:,1:]-output2[:,:,:,:-1]).pow(2)))
    diff_loss += torch.mean((output3[:,:,1:]-output3[:,:,:-1]).pow(2))+torch.mean((output3[:,:,:,1:]-output3[:,:,:,:-1]).pow(2))
    u_loss += 0.01*diff_loss

    u_loss.backward()
    output3 = output3.squeeze(1)

    optimizer.step()

    if i % 50 == 0 and i > 1:
        print(s_loss)
        imageio.imsave("train_loss_outputs/depth_"+str(i)+".png",output3[0].detach().cpu().numpy())
        coord = get_grid(output3)
        warp = F.grid_sample(xL,coord,mode="bilinear",padding_mode="border")
        imageio.imsave("train_loss_outputs/warped_"+str(i)+".png",warp[0].permute(1,2,0).detach().cpu().numpy())

