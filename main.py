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
from utils.dataloader import *
from utils.warp import just_warp
from loss import l1_loss,ssim_loss,EdgeAwareLoss
from eval_utils import end_point_error
import sys
import imageio
sys.path.append('drnseg')
sys.path.append('lib')

parser = argparse.ArgumentParser(description='stereo video main')
parser.add_argument('-superv', action='store_true')
parser.add_argument('-unsuperv', action='store_true')
parser.add_argument('--modeltype', choices=['psmnet_base','residual_drn'], default='psmnet_base')
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
parser.add_argument('--res_ckpt', default=None)
parser.add_argument('--save_to', type=str, default='models')
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--superv_batchsize', type=int, default=16)
parser.add_argument('--unsuperv_batchsize', type=int, default=8)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--lr_decay', type=float, default=1.0)
parser.add_argument('--lr_decay_cycle', type=int, default=5)
parser.add_argument('--eval_every', type=int, default=1)
parser.add_argument('--variance_masking', action='store_true')
parser.add_argument('--entropy_cutoff', type=float, default=1.6)
parser.add_argument('--lambda_edge', type=float, default=0.5)
parser.add_argument('--lambda_ssim', type=float, default=0.5)
parser.add_argument('--lambda_diff', type=float, default=0.01)
parser.add_argument('--freeze', choices=['feature_extractor'], default=None)
parser.add_argument('--debug', action='store_true')
args = parser.parse_args()

# cuda
use_cuda = torch.cuda.is_available()

# load unsupervised dataset
if args.unsuperv:
    u_trainpath = args.train_unsuperv_txt
    u_valpath = args.val_unsuperv_txt
    u_trainset = StereoSeqDataset(u_trainpath,args.seqlength,debug=args.debug)
    u_valset = StereoSeqDataset(u_valpath,args.seqlength,debug=args.debug)

    u_trainloader = DataLoader(u_trainset,batch_size=args.unsuperv_batchsize,shuffle=True,num_workers=8)
    u_evaltrainloader = DataLoader(u_trainset,batch_size=1,shuffle=False,num_workers=1)
    u_evalvalloader = DataLoader(u_valset,batch_size=1,shuffle=True,num_workers=1)

# load supervised dataset
if args.superv:
    s_trainpath = args.train_superv_txt
    if args.seqlength == 1:
        s_trainset = StereoSupervDataset(s_trainpath)
    else:
        s_trainset = StereoSeqSupervDataset(s_trainpath,args.seqlength)
    s_trainloader = DataLoader(s_trainset,batch_size=args.superv_batchsize,shuffle=True,num_workers=8)

s_valpath = args.val_superv_txt
if args.seqlength == 1:
    s_valset = StereoSupervDataset(s_valpath,to_crop=True,debug=args.debug)
else:
    s_valset = StereoSeqSupervDataset(s_valpath,args.seqlength)
if args.debug:
    s_evalvalloader = DataLoader(s_valset,batch_size=1,shuffle=True,num_workers=1)
else:
    s_evalvalloader = DataLoader(s_valset,batch_size=1,shuffle=True,num_workers=1)

if args.ckpt is not None:
    if args.modeltype == 'psmnet_base':
        ckpt_model = PSMNet(args.maxdisp,k=args.seqlength,freeze=args.freeze)

if args.modeltype == 'psmnet_base':
    model = PSMNet(args.maxdisp,k=args.seqlength,freeze=args.freeze)
elif args.modeltype == 'residual_drn':
    model = ResidualDRNet(args.maxdisp,args.ckpt,k=args.seqlength,freeze=args.freeze)

if use_cuda:
    model = nn.DataParallel(model)
    model.cuda()
    
    if args.ckpt is not None and args.modeltype == 'psmnet_base':
        ckpt_model = nn.DataParallel(ckpt_model)
        ckpt_model.cuda()

if args.res_ckpt is not None:
    model.load_state_dict(torch.load(args.res_ckpt)['state_dict'])
    start_epoch = torch.load(args.res_ckpt)['epoch']
elif args.ckpt is not None:
    if args.modeltype == 'psmnet_base':
        model.load_state_dict(torch.load(args.ckpt)['state_dict']) 
        ckpt_model.load_state_dict(torch.load(args.ckpt)['state_dict'])
        if 'epoch' in torch.load(args.ckpt):
            start_epoch = torch.load(args.ckpt)['epoch']
        else:
            start_epoch = 0
    else:
        start_epoch = 0
else:
    start_epoch = 0

optimizer = optim.Adam(model.parameters(), lr=args.lr)
edgeloss = EdgeAwareLoss()
if use_cuda:
    edgeloss = edgeloss.cuda()

def train(s_dataloader=None, u_dataloader=None, epoch=0):

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

        if iter_count > 250:
            break
          
        s_loss,u_loss = 0.0,0.0
        if iter_count < len_s_loader and not s_dataloader is None:

            optimizer.zero_grad()
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

        # unsupervised
        if iter_count < len_u_loader and not u_dataloader is None:

            optimizer.zero_grad()
            img_seq = next(u_iter)

            img_seq = Variable(img_seq,requires_grad=True)
            if use_cuda:
                img_seq = img_seq.cuda()

            if args.modeltype == 'psmnet_base':
                if args.variance_masking:
                    with torch.no_grad():
                        ent1,ent2,ent3,_,_,_ = ckpt_model(img_seq[:,0],img_seq[:,1],True)
                    output1, output2, output3 = model(img_seq[:,0],img_seq[:,1])
                else:
                    output1, output2, output3 = model(img_seq[:,0],img_seq[:,1]) # L-R input

                # get softmax for predictions
                if args.variance_masking:
                    ent1,ent2,ent3 = ent1.detach().cpu(),ent2.detach().cpu(),ent3.detach().cpu()
                    ent1,ent2,ent3 = ent1*torch.log(ent1),ent2*torch.log(ent2),ent3*torch.log(ent3)
                    ent1 = torch.where(ent1==ent1,ent1,torch.zeros(ent1.shape))
                    ent2 = torch.where(ent2==ent2,ent2,torch.zeros(ent2.shape))
                    ent3 = torch.where(ent3==ent3,ent3,torch.zeros(ent3.shape))
                    ent1,ent2,ent3 = torch.sum(-ent1,dim=1),torch.sum(-ent2,dim=1),torch.sum(-ent3,dim=1)

                    ent1_mask,ent2_mask,ent3_mask = ent1<args.entropy_cutoff,ent2<args.entropy_cutoff,ent3<args.entropy_cutoff
                    ent1_mask,ent2_mask,ent3_mask = ent1_mask.unsqueeze(1).cuda(),ent2_mask.unsqueeze(1).cuda(),ent3_mask.unsqueeze(1).cuda()

                imgL,imgR = Variable(img_seq[:,0],requires_grad=True),Variable(img_seq[:,1],requires_grad=True)
                #imgL,imgR = torch.mean(imgL,dim=1).unsqueeze(1),torch.mean(imgR,dim=1).unsqueeze(1)

                output1,output2,output3 = output1.unsqueeze(1),output2.unsqueeze(1),output3.unsqueeze(1)

                warp1 = just_warp(imgR,output1)
                warp2 = just_warp(imgR,output2)
                warp3 = just_warp(imgR,output3)
                
                # # downsampling for multiscale
                # s1_imgL = F.interpolate(imgL,scale_factor=0.25,mode='bilinear')
                # s2_imgL = F.interpolate(s1_imgL,scale_factor=0.5,mode='bilinear')
                # s3_imgL = F.interpolate(s2_imgL,scale_factor=0.5,mode='bilinear')
                # s1_imgR = F.interpolate(imgR,scale_factor=0.25,mode='bilinear')
                # s2_imgR = F.interpolate(s1_imgR,scale_factor=0.5,mode='bilinear')
                # s3_imgR = F.interpolate(s2_imgR,scale_factor=0.5,mode='bilinear')
                
                # s1_o1,s1_o2,s1_o3 = F.interpolate(output1,scale_factor=0.25,mode='bilinear'),F.interpolate(output2,scale_factor=0.25,mode='bilinear'),F.interpolate(output3,scale_factor=0.25,mode='bilinear')
                # s2_o1,s2_o2,s2_o3 = F.interpolate(s1_o1,scale_factor=0.5,mode='bilinear'),F.interpolate(s1_o2,scale_factor=0.5,mode='bilinear'),F.interpolate(s1_o3,scale_factor=0.5,mode='bilinear')
                # s3_o1,s3_o2,s3_o3 = F.interpolate(s2_o1,scale_factor=0.5,mode='bilinear'),F.interpolate(s2_o2,scale_factor=0.5,mode='bilinear'),F.interpolate(s2_o3,scale_factor=0.5,mode='bilinear')
                
                # s1_warp1,s1_warp2,s1_warp3 = just_warp(s1_imgR,s1_o1/4),just_warp(s1_imgR,s1_o2/4),just_warp(s1_imgR,s1_o3/4)
                # s2_warp1,s2_warp2,s2_warp3 = just_warp(s2_imgR,s2_o1/8),just_warp(s2_imgR,s2_o2/8),just_warp(s2_imgR,s2_o3/8)
                # s3_warp1,s3_warp2,s3_warp3 = just_warp(s3_imgR,s3_o1/16),just_warp(s3_imgR,s3_o2/16),just_warp(s3_imgR,s3_o3/16)
                
                # reverse warp

                # reverse1 = just_warp(warp1,-output1)
                # reverse2 = just_warp(warp2,-output2)
                # reverse3 = just_warp(warp3,-output3)

                # occlude1 = (reverse1+imgR).pow(2) >= 0.01*(reverse1.pow(2)+imgR.pow(2))+0.5
                # occlude2 = (reverse2+imgR).pow(2) >= 0.01*(reverse2.pow(2)+imgR.pow(2))+0.5
                # occlude3 = (reverse3+imgR).pow(2) >= 0.01*(reverse3.pow(2)+imgR.pow(2))+0.5

                #output3 = output3.unsqueeze(1)

                loss1_mask = torch.ones(imgR.shape).cuda()
                loss2_mask = torch.ones(imgR.shape).cuda()
                loss3_mask = torch.ones(imgR.shape).cuda()
                
                # # mask for downsampled images 
                # s1_mask = F.interpolate(loss1_mask,scale_factor=0.25)
                # s2_mask = F.interpolate(s1_mask,scale_factor=0.5)
                # s3_mask = F.interpolate(s2_mask,scale_factor=0.5)

                # s1_mask,s2_mask,s3_mask = s1_mask.byte(),s2_mask.byte(),s3_mask.byte()

                # get mask based on warping
                if args.variance_masking:
                    loss1_mask = just_warp(torch.ones(imgR.shape).cuda(),output1)
                    loss2_mask = just_warp(torch.ones(imgR.shape).cuda(),output2)
                    loss3_mask = just_warp(torch.ones(imgR.shape).cuda(),output3)

                # loss1_mask *= occlude1.float()
                # loss2_mask *= occlude2.float()
                # loss3_mask *= occlude3.float()

                if args.variance_masking:
                    loss1_mask *= ent1_mask.float()
                    loss2_mask *= ent2_mask.float()
                    loss3_mask *= ent3_mask.float()

                loss1_mask = torch.ceil(loss1_mask).byte()
                loss2_mask = torch.ceil(loss2_mask).byte()
                loss3_mask = torch.ceil(loss3_mask).byte()

                loss1 = l1_loss(imgL,warp1,loss1_mask) + args.lambda_edge*edgeloss(imgL,output1,loss1_mask)+args.lambda_ssim*ssim_loss(imgL,warp1,loss1_mask)
                loss2 = l1_loss(imgL,warp2,loss2_mask) + args.lambda_edge*edgeloss(imgL,output2,loss2_mask)+args.lambda_ssim*ssim_loss(imgL,warp2,loss2_mask)
                loss3 = l1_loss(imgL,warp3,loss3_mask) + args.lambda_edge*edgeloss(imgL,output3,loss3_mask)+args.lambda_ssim*ssim_loss(imgL,warp3,loss3_mask)
                
                # # downsampled loss
                # loss1 += l1_loss(s1_imgL,s1_warp1,s1_mask)+0.5*edgeloss(s1_imgL,s1_o1,s1_mask)
                # loss1 += l1_loss(s1_imgL,s1_warp2,s1_mask)+0.5*edgeloss(s1_imgL,s1_o2,s1_mask)
                # loss1 += l1_loss(s1_imgL,s1_warp3,s1_mask)+0.5*edgeloss(s1_imgL,s1_o3,s1_mask)
                # loss2 += l1_loss(s2_imgL,s2_warp1,s2_mask)+0.5*edgeloss(s2_imgL,s2_o1,s2_mask)
                # loss2 += l1_loss(s2_imgL,s2_warp2,s2_mask)+0.5*edgeloss(s2_imgL,s2_o2,s2_mask)
                # loss2 += l1_loss(s2_imgL,s2_warp3,s2_mask)+0.5*edgeloss(s2_imgL,s2_o3,s2_mask)
                # loss3 += l1_loss(s3_imgL,s3_warp1,s3_mask)+0.5*edgeloss(s3_imgL,s3_o1,s3_mask)
                # loss3 += l1_loss(s3_imgL,s3_warp2,s3_mask)+0.5*edgeloss(s3_imgL,s3_o2,s3_mask)
                # loss3 += l1_loss(s3_imgL,s3_warp3,s3_mask)+0.5*edgeloss(s3_imgL,s3_o3,s3_mask)

                diff_loss = 0.5*(torch.mean((output1[:,:,1:]-output1[:,:,:-1]).pow(2))+torch.mean((output1[:,:,:,1:]-output1[:,:,:,:-1]).pow(2)))
                diff_loss += 0.7*(torch.mean((output2[:,:,1:]-output2[:,:,:-1]).pow(2))+torch.mean((output2[:,:,:,1:]-output2[:,:,:,:-1]).pow(2)))
                diff_loss += torch.mean((output3[:,:,1:]-output3[:,:,:-1]).pow(2))+torch.mean((output3[:,:,:,1:]-output3[:,:,:,:-1]).pow(2))

                # print(l1_loss(imgL,warp2,loss3_mask))
                # print(edgeloss(imgL,output3,loss3_mask))
                # print(ssim_loss(imgL,warp3,loss3_mask))

                u_loss = (0.5*loss1 + 0.7*loss2 + loss3) + args.lambda_diff*diff_loss
                u_loss *= 1.0

            elif args.modeltype == 'residual_drn':
                if args.variance_masking:
                    with torch.no_grad():
                        ent,output = model(img_seq[:,0],img_seq[:,1],True)
                else:
                    output = model(img_seq[:,0],img_seq[:,1]) # L-R input

                if args.variance_masking:
                    ent = ent.detach().cpu()
                    ent = ent*torch.log(ent)
                    ent = torch.where(ent==ent,ent,torch.zeros(ent.shape))
                    ent = torch.sum(-ent,dim=1)

                    ent_mask = ent<args.entropy_cutoff
                    ent_mask = ent_mask.unsqueeze(1).cuda()

                imgL,imgR = Variable(img_seq[:,0]),Variable(img_seq[:,1])
                ent,output = Variable(ent,requires_grad=True),Variable(output,requires_grad=True)

                warp = just_warp(imgR,output)
                reverse = just_warp(warp,-output)

                occlude = (reverse+imgR).pow(2) >= 0.01*(reverse.pow(2)+imgR.pow(2))+0.5
                loss_mask = just_warp(torch.ones(imgR.shape).cuda(),output)

                loss_mask *= occlude.float()
                if args.variance_masking:
                    loss_mask *= ent_mask.float()
                loss_mask = loss_mask.byte()

                loss = l1_loss(imgL,warp,loss_mask)+0.5*edgeloss(imgL,output,loss_mask)+0.5*ssim_loss(imgL,warp,loss_mask)
                diff_loss = torch.mean((output[:,:,1:]-output[:,:,:-1]).pow(2))+torch.mean((output[:,:,:,1:]-output[:,:,:,:-1]).pow(2))
                
                u_loss = loss+0.0*diff_loss
                u_loss *= 1.0

            u_loss.backward()
            total_u_loss += u_loss
            total_u_n += img_seq.size(0)

            if args.debug:
                imageio.imsave("debug/img_L.png",imgL[0].permute(1,2,0).detach().cpu().numpy())
                imageio.imsave("debug/img_R.png",imgR[0].permute(1,2,0).detach().cpu().numpy())
                imageio.imsave("debug/disp_"+str(epoch)+".png",output3[0][0].detach().cpu().numpy())
                imageio.imsave("debug/warp_"+str(epoch)+".png",warp3[0].permute(1,2,0).detach().cpu().numpy())

        optimizer.step()
        iter_count += 1
        if iter_count >= term_iter: # out of data
            break

        if iter_count % 25 == 1:
            if not s_dataloader is None:
                print("training loss at iter " + str(iter_count) + " : " + str((total_s_loss/total_s_n).item()))
            if not u_dataloader is None:
                print("training loss at iter " + str(iter_count) + " : " + str((total_u_loss/total_u_n).item()))

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

        if iter_count > 100:
            break

        img_L,img_R,y = next(d_iter)
        if use_cuda:
            img_L = img_L.cuda()
            img_R = img_R.cuda()
            y = y.cuda()

        y = y.squeeze(1)
        mask = (y < args.maxdisp)*(y > 0.0)
        mask.detach_()
        
        if args.modeltype == 'psmnet_base' or args.modeltype == 'residual_drn':
            with torch.no_grad():
                output3 = model(img_L,img_R) # L-R input
            output3 = torch.squeeze(output3,1)
           
            if args.debug:
                s_loss = torch.mean(torch.abs(output3[mask]-y[mask]))
            else:
                s_loss = torch.mean((torch.abs(output3[mask]-y[mask])>3.0).float())*output3.size(0)
        
        total_loss += s_loss
        total_n += output3.size(0)
        iter_count += 1

        if iter_count % 100 == 0:
            print("validation loss at iter " + str(iter_count) + " : " + str((total_loss/total_n).item()))

    return (total_loss/total_n).item()
        
def main():

    for epoch in range(start_epoch,args.epochs):

        print("starting epoch : " + str(epoch)) 
        if epoch % args.lr_decay_cycle and epoch > 0:
            adjust_learning_rate(epoch)

        if args.superv and args.unsuperv:
            s_trainloss,epe_trainloss,u_trainloss = train(s_trainloader,u_trainloader,epoch)
            print("training supervised loss : " + str(s_trainloss) + ", epoch : " + str(epoch))
            print("training epe loss : " + str(epe_trainloss) + ", epoch : " + str(epoch))
            print("training unsupervised loss : " + str(u_trainloss) + ", epoch : " + str(epoch))
        elif args.superv:
            #print("skip training")
            s_trainloss,epe_trainloss = train(s_trainloader,None,epoch)
            print("training supervised loss : " + str(s_trainloss) + ", epoch : " + str(epoch))
            print("training epe loss : " + str(epe_trainloss) + ", epoch : " + str(epoch))
        else:
            u_trainloss = train(None,u_trainloader,epoch)
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

