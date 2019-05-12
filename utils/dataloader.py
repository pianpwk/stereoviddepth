import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import imageio
import os
import random
import sys
import utils.psmprocess as psmprocess

class StereoSeqDataset(Dataset):

    def __init__(self, datafilepath, k):
        self.filepath = datafilepath
        self.k = k
        self.preprocess = psmprocess.get_transform(augment=False)  

        datafile = open(self.filepath,'r')
        self.data = []
        while True:
            # if len(self.data) == 1:
            #     break
            sequence = []
            for i in range(self.k):
                line = datafile.readline()[:-1].split(" ")
                if len(line) < 2:
                    break
                sequence.append(line[0])
                sequence.append(line[1])
            if len(sequence) == self.k*2:
                self.data.append(sequence)
                line = datafile.readline()
            else:
                break

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence = self.data[idx]
        imgs = []
        sample_img = Image.open(sequence[0]).convert('RGB')
        w,h = sample_img.size
        ch,cw = 256,512
        x1 = random.randint(0, w-cw)
        y1 = random.randint(0, h-ch)
        # x1,y1 = 400,100
        for img in sequence:
            img = Image.open(img).convert('RGB')
            img = img.crop((x1,y1,x1+cw,y1+ch))
            img = self.preprocess(img)
            imgs.append(img)
        return torch.stack(imgs)

class StereoSeqSupervDataset(Dataset):

    def __init__(self, datafilepath, k=1):
        self.filepath = datafilepath
        self.preprocess = psmprocess.get_transform(augment=False)
        self.k = k 

        datafile = open(self.filepath,'r')
        self.images_L = []
        self.images_R = []
        self.disps = []

        seq = []
        for line in datafile:
            if line == "\n":
                self.images_L.append([scene[0] for scene in seq])
                self.images_R.append([scene[1] for scene in seq])
                self.disps.append(seq[-1][2])
                seq = []
            else:
                seq.append(line[:-1].split(" "))

    def __len__(self):
        return len(self.images_L)

    def __getitem__(self, idx):
        images_L = [Image.open(img_L).convert('RGB') for img_L in self.images_L[idx]]
        images_R = [Image.open(img_R).convert('RGB') for img_R in self.images_R[idx]]
        disp = np.array(imageio.imread(self.disps[idx]),dtype=np.float32)/256.0

        w,h = np.inf,np.inf
        for im in images_L:
            if im.size[0] < w:
                w = im.size[0]
            if im.size[1] < h:
                h = im.size[1]
        ch, cw = 256, 512
        x1 = random.randint(0, w - cw)
        y1 = random.randint(0, h - ch)

        images_L = [self.preprocess(img_L.crop((x1,y1,x1+cw,y1+ch))) for img_L in images_L]
        images_R = [self.preprocess(img_R.crop((x1,y1,x1+cw,y1+ch))) for img_R in images_R]
        disp = torch.FloatTensor(disp[y1:y1+ch,x1:x1+cw])
        
        return torch.cat(images_L,dim=0),torch.cat(images_R,dim=0),disp

class StereoSupervDataset(Dataset):

    def __init__(self, datafilepath, to_crop=True, scale_image=False, scale_type='sqrt', scale_rate=False, scale_value=1.0):
        self.filepath = datafilepath
        self.preprocess = psmprocess.get_transform(augment=False)
        self.to_crop = to_crop
        self.scale_type = scale_type
        self.scale_image = scale_image
        self.scale_rate = scale_rate
        self.scale_value = scale_value

        datafile = open(self.filepath,'r')
        self.images_L = []
        self.images_R = []
        self.disps = []

        for line in datafile:
            line = line[:-1].split(" ")
            self.images_L.append(line[0])
            self.images_R.append(line[1])
            self.disps.append(line[2])

    def __len__(self):
        return len(self.images_L)

    def __getitem__(self, idx):
        
        disp = np.array(imageio.imread(self.disps[idx]),dtype=np.float32)/256.0
        if self.to_crop:
            image_L = Image.open(self.images_L[idx]).convert('RGB')
            image_R = Image.open(self.images_R[idx]).convert('RGB')
            w, h = image_L.size
            ch, cw = 256, 512
            x1 = random.randint(0, w - cw)
            y1 = random.randint(0, h - ch)
            image_L = self.preprocess(image_L.crop((x1,y1,x1+cw,y1+ch)))
            image_R = self.preprocess(image_R.crop((x1,y1,x1+cw,y1+ch)))
            disp = disp[y1:y1+ch,x1:x1+cw]
            disp = torch.FloatTensor(disp)

            return image_L,image_R,disp

        else:
            image_L = imageio.imread(self.images_L[idx])
            image_R = imageio.imread(self.images_R[idx])
            if self.scale_image:
                if self.scale_type == 'sqrt':
                    image_L = np.sqrt(image_L)
                    image_R = np.sqrt(image_R)
                elif self.scale_type == 'cbrt':
                    image_L = np.cbrt(image_L)
                    image_R = np.cbrt(image_R)
            elif self.scale_rate:
                image_L = image_L/self.scale_value
                image_R = image_R/self.scale_value
            image_L = image_L/np.max(image_L)*255
            image_R = image_R/np.max(image_R)*255

            oh,ow = image_L.shape[0],image_L.shape[1]
            image_L = np.pad(image_L,((0,384-oh),(0,1248-ow),(0,0)),mode="constant",constant_values=0)
            image_R = np.pad(image_R,((0,384-oh),(0,1248-ow),(0,0)),mode="constant",constant_values=0)
            image_L = self.preprocess(Image.fromarray(np.uint8(image_L)))
            image_R = self.preprocess(Image.fromarray(np.uint8(image_R)))
            disp = np.pad(disp,((0,384-oh),(0,1248-ow)),mode="constant",constant_values=0)
            disp = torch.FloatTensor(disp)

            return image_L,image_R,disp,oh,ow
