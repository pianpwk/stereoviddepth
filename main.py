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
from eval import end_point_error
import sys
import imageio
sys.path.append('drnseg')
sys.path.append('lib')

import torch
import gc

parser = argparse.ArgumentParser(description='stereo video main')
parser.add_argument('-superv', action='store_true')
parser.add_argument('-unsuperv', action='store_true')
parser.add_argument('--modeltype', choices=['psmnet_base'], default='psmnet_base')
parser.add_argument('--maxdisp', type=int, default=192,
                    help='maximum disparity')

parser.add_argument('--train_superv_txt', type=str, default='data/train_supervised.txt',
                    help='txt file for train S')
parser.add_argument('--val_superv_txt', type=str, default='data/val_supervised.txt',
                    help='txt file for val S')
parser.add_argument('--train_unsuperv_txt', type=str, default='data/train_unsupervised.txt',
                    help='txt file for train U')
parser.add_argument('--val_unsuperv_txt', type=str, default='data/train_unsupervised.txt',
                    help='txt file for val U')

parser.add_argument('--seqlength', type=int, default=3,
                    help='sequence length')
parser.add_argument('--ckpt', default=None,
                    help='checkpoint model')
parser.add_argument('--save_to', type=str, default='models')
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--superv_batchsize', type=int, default=16)
parser.add_argument('--unsuperv_batchsize', type=int, default=8)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--lr_decay', type=float, default=1.0)
parser.add_argument('--lr_decay_cycle', type=int, default=5)
parser.add_argument('--eval_every', type=int, default=1)
args = parser.parse_args()

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
    c = torch.clamp(c*mult-1,-1,1)
    return c

# load unsupervised dataset
if args.unsuperv:
    u_trainpath = args.train_unsuperv_txt
    u_valpath = args.val_unsuperv_txt
    u_trainset = StereoSeqDataset(u_trainpath,args.seqlength)
    u_valset = StereoSeqDataset(u_valpath,args.seqlength)

    u_trainloader = DataLoader(u_trainset,batch_size=args.unsuperv_batchsize,shuffle=True,num_workers=8)
    u_evaltrainloader = DataLoader(u_trainset,batch_size=1,shuffle=False,num_workers=8)
    u_evalvalloader = DataLoader(u_valset,batch_size=1,shuffle=True,num_workers=8)

# load supervised dataset
if args.superv:
    s_trainpath = args.train_superv_txt
    s_trainset = StereoSupervDataset(s_trainpath)
    s_trainloader = DataLoader(s_trainset,batch_size=args.superv_batchsize,shuffle=True,num_workers=8)

s_valpath = args.val_superv_txt
s_valset = StereoSupervDataset(s_valpath)
s_evalvalloader = DataLoader(s_valset,batch_size=1,shuffle=True,num_workers=8)

model = PSMNet(args.maxdisp)

if use_cuda:
    model = nn.DataParallel(model)
    model.cuda()

if args.ckpt is not None:
    model.load_state_dict(torch.load(args.ckpt)['state_dict'])

optimizer = optim.Adam(model.parameters(), lr=args.lr)
edgeloss = EdgeAwareLoss()
if use_cuda:
    edgeloss = edgeloss.cuda()

def train(s_dataloader=None, u_dataloader=None):

    model.train()

    total_u_loss = 0.0
    total_u_n = 0
    total_s_loss = 0.0
    total_s_n = 0
    total_epe_loss = 0.0
    total_tpe_loss = 0.0
 
    iter_count = 0
    if not s_dataloader is None:
        len_s_loader = len(s_dataloader)
        s_iter = iter(s_dataloader)
    else:
        len_s_loader = 0
    if not u_dataloader is None:
        len_u_loader = len(u_dataloader)
        u_iter = iter(u_dataloader)
    else:
        len_u_loader = 0

    if not s_dataloader is None and not u_dataloader is None:
        term_iter = min(len_s_loader,len_u_loader)
    elif not s_dataloader is None:
        term_iter = len_s_loader
    else:
        term_iter = len_u_loader

    print(len_s_loader,len_u_loader)
    while True:
        
        optimizer.zero_grad()   
        s_loss,u_loss = 0.0,0.0
        if iter_count < len_s_loader and not s_dataloader is None:
            img_L,img_R,y = next(s_iter)

            if use_cuda:
                img_L = img_L.cuda()
                img_R = img_R.cuda()
                y = y.cuda()

            y = y.squeeze(1)
            mask = (y < args.maxdisp)*(y > 0.0)
            mask.detach_()

            if args.modeltype == 'psmnet_base':
                output1, output2, output3 = model(img_L,img_R) # L-R input
                output1 = torch.squeeze(output1,1)
                output2 = torch.squeeze(output2,1)
                output3 = torch.squeeze(output3,1)

                s_loss = 0.5*F.smooth_l1_loss(output1[mask], y[mask], size_average=True) + 0.7*F.smooth_l1_loss(output2[mask], y[mask], size_average=True) + F.smooth_l1_loss(output3[mask], y[mask], size_average=True)
                epe_loss = end_point_error(output3,y,mask)
                tpe_loss = torch.mean((torch.abs(output3[mask]-y[mask])>3.0).float())*output3.size(0)

            s_loss.backward()

            total_s_n += output1.size(0)
            total_s_loss += s_loss
            total_epe_loss += epe_loss
            total_tpe_loss += tpe_loss
        if iter_count < len_u_loader and not u_dataloader is None:
            img_seq = next(u_iter)

            if use_cuda:
                img_seq = img_seq.cuda()

            if args.modeltype == 'psmnet_base':
                output1, output2, output3 = model(img_seq[:,0],img_seq[:,1]) # L-R input
                output1 = torch.squeeze(output1,1)
                output2 = torch.squeeze(output2,1)
                output3 = torch.squeeze(output3,1)
                coord1 = get_grid(output1)
                coord2 = get_grid(output2)
                coord3 = get_grid(output3)

                warp1 = F.grid_sample(img_seq[:,0],coord1,padding_mode="border")
                warp2 = F.grid_sample(img_seq[:,0],coord2,padding_mode="border")
                warp3 = F.grid_sample(img_seq[:,0],coord3,padding_mode="border")

                output1,output2,output3 = output1.unsqueeze(1),output2.unsqueeze(1),output3.unsqueeze(1)
                #output3 = output3.unsqueeze(1)

                loss1 = 0.5*l1_loss(img_seq[:,1],warp1) + 0.5*ssim_loss(img_seq[:,1],warp1) + edgeloss(img_seq[:,1],output1)
                loss2 = 0.5*l1_loss(img_seq[:,1],warp2) + 0.5*ssim_loss(img_seq[:,1],warp2) + edgeloss(img_seq[:,1],output2)
                loss3 = 0.5*l1_loss(img_seq[:,1],warp3) + 0.5*ssim_loss(img_seq[:,1],warp3) + edgeloss(img_seq[:,1],output3)
                u_loss = (0.5*loss1 + 0.7*loss2 + loss3)/(256.0*512.0)
                #u_loss = loss1+loss2+loss3/(256.0*512.0)
                # do computation for unsupervised reconstruction, and compute loss

            u_loss.backward()

            total_u_loss += u_loss
            total_u_n += img_seq.size(0)
        optimizer.step()
        iter_count += 1
        if iter_count >= term_iter: # out of data
            break
    if not s_dataloader is None and not u_dataloader is None:
        return (total_s_loss/total_s_n).item(),(total_epe_loss/total_s_n).item(),(total_u_loss/total_u_n).item()
    elif s_dataloader is None:
        return (total_u_loss/total_u_n).item()
    else:
        return (total_s_loss/total_s_n).item(),(total_epe_loss/total_s_n).item()

def adjust_learning_rate(epoch):
    lr = args.lr * (args.lr_decay ** int(epoch/args.lr_decay_cycle))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def eval_supervised(dataloader): # only takes in supervised loader

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
        
def main():

    for epoch in range(args.epochs):

        if epoch % args.lr_decay_cycle and epoch > 0:
            adjust_learning_rate(epoch)

        if args.superv and args.unsuperv:
            s_trainloss,epe_trainloss,u_trainloss = train(s_trainloader,u_trainloader)
            print("training supervised loss : " + str(s_trainloss) + ", epoch : " + str(epoch))
            print("training epe loss : " + str(epe_trainloss) + ", epoch : " + str(epoch))
            print("training unsupervised loss : " + str(u_trainloss) + ", epoch : " + str(epoch))
        elif args.superv:
            #print("skip training")
            s_trainloss,epe_trainloss = train(s_trainloader,None)
            print("training supervised loss : " + str(s_trainloss) + ", epoch : " + str(epoch))
            print("training epe loss : " + str(epe_trainloss) + ", epoch : " + str(epoch))
        else:
            u_trainloss = train(None,u_trainloader)
            print("training unsupervised loss : " + str(u_trainloss) + ", epoch : " + str(epoch))

        if epoch % args.eval_every == 0:
             valloss = eval_supervised(s_evalvalloader)
             print("validation 3 pixel error : " + str(valloss) + ", epoch : " + str(epoch))
 
             savefilename = args.save_to+'/checkpoint_'+str(epoch)+'.tar'
             torch.save({
                 'epoch': epoch,
                 'state_dict': model.state_dict(),
                 'val_loss': valloss,
             }, savefilename)

if __name__ == '__main__':
   main()

