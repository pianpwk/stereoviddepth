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
import kitti_object
import kitti_util

# cal = kitti_util.Calibration('dorn_calib/'+str(num)+'.txt')
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
    
    warped = F.grid_sample(image,c,padding_mode="border")
    return warped

def assign_img_for_pc(cloud,img,cal):
    img_height, img_width, _ = img.shape
    _, _, img_fov_inds = get_lidar_in_image_fov(cloud[:, :3], cal, 0, 0, img_width, img_height, True)
    cloud = cloud[img_fov_inds]

    depth_map = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
    ptc_image = cal.project_velo_to_image3(cloud[:, :3])
    ptc_2d = np.floor(ptc_image[:, :2]).astype(np.int32)
    for i in range(ptc_2d.shape[0]):
        depth_map[ptc_2d[i, 1], ptc_2d[i, 0]] = ptc_image[i, 2]

    return depth_map
