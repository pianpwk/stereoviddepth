import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset
import numpy as np
from PIL import Image
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
        for img in sequence:
            img = Image.open(img).convert('RGB')
            w, h = img.size
            ch, cw = 256, 512
            x1 = random.randint(0, w - cw)
            y1 = random.randint(0, h - ch)
            img = img.crop((x1,y1,x1+cw,y1+ch))
            img = self.preprocess(img)
            imgs.append(img)
        return torch.stack(sequence)

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
        disp = Image.open(self.disps[idx]).convert('RGB')
        w, h = image_L.size
        ch, cw = 256, 512
        x1 = random.randint(0, w - cw)
        y1 = random.randint(0, h - ch)

        image_L = self.preprocess(image_L.crop((x1,y1,x1+cw,y1+ch)))
        image_R = self.preprocess(image_R.crop((x1,y1,x1+cw,y1+ch)))
        disp = self.preprocess(disp.crop((x1,y1,x1+cw,y1+ch)))

        return image_L,image_R,disp
