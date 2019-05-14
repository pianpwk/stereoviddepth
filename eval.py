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
from lib.model import DRNSegment,PSMNet,ResidualDRNet
from utils.dataloader import StereoSeqDataset,StereoSupervDataset
from utils.warp import just_warp
from loss import l1_loss,ssim_loss,EdgeAwareLoss
from eval_utils import end_point_error
import sys
import imageio
sys.path.append('drnseg')
sys.path.append('lib')

parser = argparse.ArgumentParser(description='evaluation scheme')
parser.add_argument('--modeltype', choices=['psmnet_base','residual_drn'], default='psmnet_base')
parser.add_argument('--maxdisp', type=int, default=192,
                    help='maximum disparity')

parser.add_argument('--val_txt', type=str, default='data/val_supervised.txt',
                    help='txt file for val')
parser.add_argument('--seqlength', type=int, default=3,
                    help='sequence length')
parser.add_argument('--ckpt', default=None,
                    help='checkpoint model')
parser.add_argument('--res_ckpt', default=None)
parser.add_argument('--sample_output', action='store_true')
parser.add_argument('--eval_type', choices=['disparity','depth'], default='disparity')
parser.add_argument('--scale_image', action='store_true')
parser.add_argument('--scale_type', choices=['sqrt','cbrt'], default='sqrt')
parser.add_argument('--scale_rate', action='store_true')
parser.add_argument('--scale_value', type=float, default=1.0)
args = parser.parse_args()

# cuda
use_cuda = torch.cuda.is_available()

valpath = args.val_txt
valset = StereoSupervDataset(valpath,to_crop=False)
evalvalloader = DataLoader(valset,batch_size=4,shuffle=False,num_workers=4)

if args.modeltype == 'psmnet_base':
    model = PSMNet(args.maxdisp,k=args.seqlength)
elif args.modeltype == 'residual_drn':
    model = ResidualDRNet(args.maxdisp,args.ckpt,k=args.seqlength)

if use_cuda:
    model = nn.DataParallel(model)
    model.cuda()

if args.res_ckpt is not None:
    model.load_state_dict(torch.load(args.res_ckpt)['state_dict'])
elif args.ckpt is not None:
    model.load_state_dict(torch.load(args.ckpt)['state_dict'])

def eval(dataloader): # only takes in supervised loader

    model.eval()

    total_loss = 0.0
    total_n = 0
    depth_errors = [[] for i in range(16)]

    iter_count = 0
    len_iter = len(dataloader)
    d_iter = iter(dataloader)
    while iter_count < len_iter:

        if iter_count > 250:
            break
        print(iter_count)
        img_L,img_R,y,oh,ow = next(d_iter)
        if use_cuda:
            img_L = img_L.cuda()
            img_R = img_R.cuda()
            y = y.cuda()

        if args.scale_rate:
            img_L = img_L/args.scale_value
            img_R = img_R/args.scale_value

        y = y.squeeze(1)
        mask = (y < args.maxdisp)*(y > 0.0)
        mask.detach_()
        
        if args.modeltype == 'psmnet_base' or args.modeltype == 'residual_drn':
            with torch.no_grad():
                output3 = model(img_L,img_R) # L-R input
            output3 = torch.squeeze(output3,1)

            if args.sample_output:
                warp3 = just_warp(img_R,output3)
                imageio.imsave("sample_outputs/"+str(iter_count)+"_imgL.png",img_L[0].permute(1,2,0).detach().cpu().numpy())
                imageio.imsave("sample_outputs/"+str(iter_count)+"_imgR.png",img_R[0].permute(1,2,0).detach().cpu().numpy())
                imageio.imsave("sample_outputs/"+str(iter_count)+"_warped.png",warp3[0].permute(1,2,0).detach().cpu().numpy())
                np.save("sample_outputs/"+str(iter_count)+"_depth.npy",output3[0].detach().cpu().numpy())
 
            if args.eval_type == 'disparity':
                s_loss = torch.mean((torch.abs(output3[mask]-y[mask])>3.0).float())*output3.size(0)
            elif args.eval_type == 'depth':
                output3,y = output3.squeeze(1).detach().cpu(),y.detach().cpu()
                y = torch.where(y>0.0,0.54*721/y,torch.zeros(y.shape))
                output3 = 0.54*721/output3
                for i in range(16):
                    depth_mask = (y > i*5)*(y < (i+1)*5)
                    errors = torch.abs(output3[depth_mask]-y[depth_mask])
                    depth_errors[i].append(errors)
        
        if args.eval_type == 'disparity':
            total_loss += s_loss
            total_n += output3.size(0)   
        iter_count += 1

    if args.eval_type == 'disparity':
        print("total type loss : " + str((total_loss/total_n).item()))
    elif args.eval_type == 'depth':
        for i in range(16):
            errors = np.concatenate(depth_errors[i])
            print("median depth error for range " + str(5*i) + " - " + str(5*i+5) + " : " + str(np.median(errors)))

if __name__ == '__main__':
    eval(evalvalloader)
