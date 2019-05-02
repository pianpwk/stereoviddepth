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
                self.disps.append(scene[-1][2])
                seq = []
            else:
                seq.append(line[:-1].split(" "))

    def __len__(self):
        return len(self.images_L)

    def __getitem__(self, idx):
        images_L = [Image.open(img_L).convert('RGB') for img_L in self.images_L[idx]]
        images_R = [Image.open(img_R).convert('RGB') for img_R in self.images_R[idx]]
        disp = np.array(imageio.imread(self.disps[idx]),dtype=np.float32)/256.0

        w, h = images_L[0].size
        ch, cw = 256, 512
        x1 = random.randint(0, w - cw)
        y1 = random.randint(0, h - ch)

        images_L = [self.preprocess(img_L.crop((x1,y1,x1+cw,y1+ch))) for img_L for images_L]
        images_R = [self.preprocess(img_R.crop((x1,y1,x1+cw,y1+ch))) for img_R for images_R]
        disp = torch.FloatTensor(disp[y1:y1+ch,x1:x1+cw])
        
        return torch.cat(images_L,dim=0),torch.cat(images_R,dim=0),disp

class StereoSupervDataset(Dataset):

    def __init__(self, datafilepath):
        self.filepath = datafilepath
        self.preprocess = psmprocess.get_transform(augment=False)

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
        image_L = Image.open(self.images_L[idx]).convert('RGB')
        image_R = Image.open(self.images_R[idx]).convert('RGB')
        disp = np.array(imageio.imread(self.disps[idx]),dtype=np.float32)/256.0
        w, h = image_L.size
        ch, cw = 256, 512
        x1 = random.randint(0, w - cw)
        y1 = random.randint(0, h - ch)

        image_L = self.preprocess(image_L.crop((x1,y1,x1+cw,y1+ch)))
        image_R = self.preprocess(image_R.crop((x1,y1,x1+cw,y1+ch)))
        disp = disp[y1:y1+ch,x1:x1+cw]
        disp = torch.FloatTensor(disp)

        return image_L,image_R,disp