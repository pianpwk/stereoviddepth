import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
sys.path.append('drnseg')

from drnseg.drn import DRNSeg

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
