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
from loss import l1_loss,ssim_loss,EdgeAwareLoss
from eval_utils import end_point_error
import sys
import imageio
sys.path.append('drnseg')
sys.path.append('lib')

parser = argparse.ArgumentParser(description='evaluation scheme')
parser.add_argument('--modeltype', choices['psmnet_base'], default='psmnet_base')
parser.add_argument('--maxdisp', type=int, default=192,
                    help='maximum disparity')

parser.add_argument('--val_txt', type=str, default='data/val_supervised.txt',
                    help='txt file for val')
parser.add_argument('--seqlength', type=int, default=3,
                    help='sequence length')
parser.add_argument('--ckpt', default=None,
                    help='checkpoint model')
args = parser.parse_args()

# cuda
use_cuda = torch.cuda.is_available()

valpath = args.val_txt
valset = StereoSupervDataset(valpath)
evalvalloader = DataLoader(valset,batch_size=8,shuffle=False,num_workers=8)

model = PSMNet(args.maxdisp)

if use_cuda:
    model = nn.DataParallel(model)
    model.cuda()

if args.ckpt is not None:
    model.load_state_dict(torch.load(args.ckpt)['state_dict'])

def eval(dataloader): # only takes in supervised loader

    model.eval()

    total_loss = 0.0
    total_n = 0

    iter_count = 0
    len_iter = len(dataloader)
    d_iter = iter(dataloader)
    while iter_count < len_iter:

        img_L,img_R,y = next(d_iter)
        if use_cuda:
            img_L = img_L.cuda()
            img_R = img_R.cuda()
            y = y.cuda()

        y = y.squeeze(1)
        mask = (y < args.maxdisp)*(y > 0.0)
        mask.detach_()
        
        if args.modeltype == 'psmnet_base':
            with torch.no_grad():
                output3 = model(img_L,img_R) # L-R input
            output3 = torch.squeeze(output3,1)

            s_loss = torch.mean((torch.abs(output3[mask]-y[mask])>3.0).float())*output3.size(0)
        
        total_loss += s_loss
        total_n += output3.size(0)   
        iter_count += 1

    return (total_loss/total_n).item()

if __name__ == '__main__':
    print("total tpe loss : " + str(eval(evalvalloader)))