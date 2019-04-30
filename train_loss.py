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
imgL_0 = get_image('../some_kitti/0_2.png',x1,y1)
imgR_0 = get_image('../some_kitti/0_3.png',x1,y1)
disp_0 = get_disp('../some_kitti/0_depth.png',x1,y1)

x1 = random.randint(0, 1230 - cw)
y1 = random.randint(0, 360 - ch)
imgL_1 = get_image('../some_kitti/3_2.png',x1,y1)
imgR_1 = get_image('../some_kitti/3_3.png',x1,y1)
disp_1 = get_disp('../some_kitti/3_depth.png',x1,y1)

x1 = random.randint(0, 1230 - cw)
y1 = random.randint(0, 360 - ch)
imgL_2 = get_image('../some_kitti/7_2.png',x1,y1)
imgR_2 = get_image('../some_kitti/7_3.png',x1,y1)
disp_2 = get_disp('../some_kitti/7_depth.png',x1,y1)

x1 = random.randint(0, 1230 - cw)
y1 = random.randint(0, 360 - ch)
imgL_3 = get_image('../some_kitti/9_2.png',x1,y1)
imgR_3 = get_image('../some_kitti/9_3.png',x1,y1)
disp_3 = get_disp('../some_kitti/9_depth.png',x1,y1)

x1 = random.randint(0, 1230 - cw)
y1 = random.randint(0, 360 - ch)
imgL_4 = get_image('../some_kitti/10_2.png',x1,y1)
imgR_4 = get_image('../some_kitti/10_3.png',x1,y1)
disp_4 = get_disp('../some_kitti/10_depth.png',x1,y1)

# train with 1 image

for i in range(1):
    
    optimizer.zero_grad()
    
    xL = imgL_0.unsqueeze(0)
    xR = imgR_0.unsqueeze(0)
    y = disp_0.unsqueeze(0)
    
    y = y.squeeze(1)
    mask = (y < 192)*(y > 0.0)
    mask.detach_()
    
    output1, output2, output3 = model(xL,xR) # L-R input
    output1 = torch.squeeze(output1,1)
    output2 = torch.squeeze(output2,1)
    output3 = torch.squeeze(output3,1)
    

