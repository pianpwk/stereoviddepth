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
#import utils.kitti_object
#import utils.kitti_util

# cal = kitti_util.Calibration('dorn_calib/'+str(num)+'.txt')

use_cuda = torch.cuda.is_available()

def just_warp(img, disp):
    B,C,H,W = img.size()
    xx = torch.arange(0, W).view(1,-1).repeat(H,1)
    yy = torch.arange(0,H).view(-1,1).repeat(1,W)
    xx = xx.view(1,1,H,W).repeat(B,1,1,1).float()
    yy = yy.view(1,1,H,W).repeat(B,1,1,1).float()
    if use_cuda:
        xx,yy = xx.cuda(),yy.cuda()

    xx = xx-disp
    xx = 2.0*xx/max(W-1,1) - 1.0
    yy = 2.0*yy/max(H-1,1) - 1.0
    grid = torch.cat((xx,yy),1).float()

    if use_cuda:
        grid = grid.cuda()
    vgrid = Variable(grid,requires_grad=True)
    vgrid = vgrid.permute(0,2,3,1)

    output = F.grid_sample(img, vgrid, mode='bilinear', padding_mode='zeros')
    return output

def warp(image,disp):

    image = torch.FloatTensor(image).permute(2,0,1).unsqueeze(0)
    disp = torch.FloatTensor(disp).unsqueeze(0)

    sh,sw = image.size(2),image.size(3)
    ch,cw = torch.Tensor(range(sh)).unsqueeze(1).repeat(1,sw),torch.Tensor(range(sw)).unsqueeze(0).repeat(sh,1)
    coord_matrix = torch.cat((cw.unsqueeze(-1),ch.unsqueeze(-1)),dim=-1)
    mult = torch.ones((sh,sw,2))
    mult[:,:,0] /= (sw-1)/2.0
    mult[:,:,1] /= (sh-1)/2.0

    c = coord_matrix.view(1,sh,sw,2).repeat(disp.size(0),1,1,1)
    c += torch.cat((disp.unsqueeze(-1),torch.zeros(disp.size(0),sh,sw,1)),dim=-1)
    c = torch.clamp(c*mult-1,-1,1)
    
    warped = F.grid_sample(image,c,mode='bilinear',padding_mode="zeros")
    return warped

def assign_img_for_pc(cloud,img,cal):
    height,width = img.shape[0],img.shape[1]

    img = np.zeros([height, width])
    imgfov_pc_velo, pts_2d, fov_inds = kitti_object.get_lidar_in_image_fov(cloud,
                                                              cal, 0, 0, width-1, height-1, True)
    imgfov_pts_2d = pts_2d[fov_inds,:]
    imgfov_pc_rect = cal.project_velo_to_rect(imgfov_pc_velo)
    depth_map = np.zeros((height, width)) - 1
    imgfov_pts_2d = np.round(imgfov_pts_2d).astype(int)
    for i in range(imgfov_pts_2d.shape[0]):
        depth = imgfov_pc_rect[i,2]
        img[imgfov_pts_2d[i, 1], imgfov_pts_2d[i, 0]] = depth
    return img
