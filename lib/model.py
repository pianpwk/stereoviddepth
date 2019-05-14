import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.autograd import Variable
import math
from lib.psmsubmodule import *
from utils.warp import just_warp
import numpy as np
import os
import sys
sys.path.append('drnseg')

from drnseg.segment import DRNSeg

class DRNSegment(nn.Module):

    def __init__(self, model_name, model_path):
        super(DRNSegment, self).__init__()
        self.model_name = model_name

        self.model = DRNSeg(model_name,19,pretrained_model=None,pretrained=False)
        self.model.load_state_dict(torch.load(model_path))

    def forward(self, x):
        preds = torch.max(self.model(x)[0],1)
        dynamic = (preds==11)+(preds==12)+(preds==13)+(preds==14)+(preds==15)+(preds==16)+(preds==17)+(preds==18)
        static = 1-dynamic
        return dynamic, static

class hourglass(nn.Module):
    def __init__(self, inplanes):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(convbn_3d(inplanes, inplanes*2, kernel_size=3, stride=2, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn_3d(inplanes*2, inplanes*2, kernel_size=3, stride=1, pad=1)

        self.conv3 = nn.Sequential(convbn_3d(inplanes*2, inplanes*2, kernel_size=3, stride=2, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(convbn_3d(inplanes*2, inplanes*2, kernel_size=3, stride=1, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv5 = nn.Sequential(nn.ConvTranspose3d(inplanes*2, inplanes*2, kernel_size=3, padding=1, output_padding=1, stride=2,bias=False),
                                   nn.BatchNorm3d(inplanes*2)) #+conv2

        self.conv6 = nn.Sequential(nn.ConvTranspose3d(inplanes*2, inplanes, kernel_size=3, padding=1, output_padding=1, stride=2,bias=False),
                                   nn.BatchNorm3d(inplanes)) #+x

    def forward(self, x ,presqu, postsqu):
        
        out  = self.conv1(x) #in:1/4 out:1/8
        pre  = self.conv2(out) #in:1/8 out:1/8
        if postsqu is not None:
           pre = F.relu(pre + postsqu, inplace=True)
        else:
           pre = F.relu(pre, inplace=True)

        out  = self.conv3(pre) #in:1/8 out:1/16
        out  = self.conv4(out) #in:1/16 out:1/16

        if presqu is not None:
           post = F.relu(self.conv5(out)+presqu, inplace=True) #in:1/16 out:1/8
        else:
           post = F.relu(self.conv5(out)+pre, inplace=True) 

        out  = self.conv6(post)  #in:1/8 out:1/4

        return out, pre, post

class PSMNet(nn.Module):
    def __init__(self, maxdisp, k=1, freeze=None):
        super(PSMNet, self).__init__()
        self.maxdisp = maxdisp

        self.feature_extraction = feature_extraction(k)

        self.dres0 = nn.Sequential(convbn_3d(64, 32, 3, 1, 1),
                                     nn.ReLU(inplace=True),
                                     convbn_3d(32, 32, 3, 1, 1),
                                     nn.ReLU(inplace=True))

        self.dres1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1)) 

        self.dres2 = hourglass(32)

        self.dres3 = hourglass(32)

        self.dres4 = hourglass(32)

        self.classif1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1,bias=False))

        self.classif2 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1,bias=False))

        self.classif3 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1,bias=False))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1]*m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

        if freeze == 'feature_extractor':
            for p in self.dres0.parameters():
                p.requires_grad = False
            for p in self.dres1.parameters():
                p.requires_grad = False
            for p in self.dres2.parameters():
                p.requires_grad = False
            for p in self.dres3.parameters():
                p.requires_grad = False
            for p in self.dres4.parameters():
                p.requires_grad = False
            for p in self.classif1.parameters():
                p.requires_grad = False
            for p in self.classif2.parameters():
                p.requires_grad = False
            for p in self.classif3.parameters():
                p.requires_grad = False

    def forward(self, left, right, get_softmax=False):

        refimg_fea     = self.feature_extraction(left)
        targetimg_fea  = self.feature_extraction(right)

        #matching
        cost = Variable(torch.FloatTensor(refimg_fea.size()[0], refimg_fea.size()[1]*2, int(self.maxdisp/4),  refimg_fea.size()[2],  refimg_fea.size()[3]).zero_()).cuda()

        for i in range(int(self.maxdisp/4)):
            if i > 0 :
             cost[:, :refimg_fea.size()[1], i, :,i:]   = refimg_fea[:,:,:,i:]
             cost[:, refimg_fea.size()[1]:, i, :,i:] = targetimg_fea[:,:,:,:-i]
            else:
             cost[:, :refimg_fea.size()[1], i, :,:]   = refimg_fea
             cost[:, refimg_fea.size()[1]:, i, :,:]   = targetimg_fea
        cost = cost.contiguous()

        cost0 = self.dres0(cost)
        cost0 = self.dres1(cost0) + cost0

        out1, pre1, post1 = self.dres2(cost0, None, None) 
        out1 = out1+cost0

        out2, pre2, post2 = self.dres3(out1, pre1, post1) 
        out2 = out2+cost0

        out3, pre3, post3 = self.dres4(out2, pre1, post2) 
        out3 = out3+cost0

        cost1 = self.classif1(out1)
        cost2 = self.classif2(out2) + cost1
        cost3 = self.classif3(out3) + cost2

        if self.training:
            cost1 = F.upsample(cost1, [self.maxdisp,left.size()[2],left.size()[3]], mode='trilinear')
            cost2 = F.upsample(cost2, [self.maxdisp,left.size()[2],left.size()[3]], mode='trilinear')

            cost1 = torch.squeeze(cost1,1)
            pred1 = F.softmax(cost1,dim=1)
            pred1_ = disparityregression(self.maxdisp)(pred1)

            cost2 = torch.squeeze(cost2,1)
            pred2 = F.softmax(cost2,dim=1)
            pred2_ = disparityregression(self.maxdisp)(pred2)

        cost3 = F.upsample(cost3, [self.maxdisp,left.size()[2],left.size()[3]], mode='trilinear')
        cost3 = torch.squeeze(cost3,1)
        pred3 = F.softmax(cost3,dim=1)
	#For your information: This formulation 'softmax(c)' learned "similarity" 
	#while 'softmax(-c)' learned 'matching cost' as mentioned in the paper.
	#However, 'c' or '-c' do not affect the performance because feature-based cost volume provided flexibility.
        pred3_ = disparityregression(self.maxdisp)(pred3)

        if get_softmax:
            if self.training:
                return pred1, pred2, pred3, pred1_, pred2_, pred3_
            else:
                return pred3, pred3_
        else:
            if self.training:
                return pred1_, pred2_, pred3_
            else:
                return pred3_

class ResidualDRNet(nn.Module):
    def __init__(self, maxdisp, ckpt, k=1, freeze=None):
        super(ResidualDRNet, self).__init__()

        self.maxdisp = maxdisp
        self.psmnet = PSMNet(maxdisp,k,freeze)

        model = nn.DataParallel(model)
        model.load_state_dict(torch.load(ckpt)['state_dict'])
        model.eval()

        self.i5 = nn.Sequential(nn.Conv2d(6,16,3,1,1),
                                nn.BatchNorm2d(16),
                                nn.ReLU(inplace=True))
        self.i6 = nn.Sequential(nn.Conv2d(2,16,3,1,1),
                                nn.BatchNorm2d(16),
                                nn.ReLU(inplace=True))
        self.i8 = nn.Sequential(nn.Conv2d(32,32,3,1,1,1),
                                nn.BatchNorm2d(32),
                                nn.ReLU(inplace=True))
        self.i9 = nn.Sequential(nn.Conv2d(32,32,3,1,2,2),
                                nn.BatchNorm2d(32),
                                nn.ReLU(inplace=True))
        self.i10 = nn.Sequential(nn.Conv2d(32,32,3,1,4,4),
                                nn.BatchNorm2d(32),
                                nn.ReLU(inplace=True))
        self.i11 = nn.Sequential(nn.Conv2d(32,32,3,1,8,8),
                                nn.BatchNorm2d(32),
                                nn.ReLU(inplace=True))
        self.i12 = nn.Sequential(nn.Conv2d(32,32,3,1,1,1),
                                nn.BatchNorm2d(32),
                                nn.ReLU(inplace=True))
        self.i13 = nn.Sequential(nn.Conv2d(32,32,3,1,1,1),
                                nn.BatchNorm2d(32),
                                nn.ReLU(inplace=True))
        self.i14 = nn.Conv2d(32,1,3,1,1)

    def forward(self,imgL,imgR,get_softmax=False):
        with torch.no_grad():
            if get_softmax:
                ent,disp = self.psmnet(imgL,imgR,True)
            else:
                disp = self.psmnet(imgL,imgR,False)

        x0 = just_warp(imgR,disp)-imgL # 1
        x0 = torch.cat((x0,imgL),dim=1) # 2
        x1 = just_warp(just_warp(imgL,-disp),disp)-disp #3
        x1 = torch.cat((x1,disp),dim=1) # 4
        x0 = self.i5(x0)
        x1 = self.i6(x1)
        x = torch.cat((x0,x1),dim=1) # 7
        x = self.i8(x)
        x = self.i9(x)
        x = self.i10(x)
        x = self.i11(x)
        x = self.i12(x)
        x = self.i13(x)
        x = self.i14(x)

        if get_softmax:
            return ent,x
        else:
            return x
