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
import sys

def end_point_error(pred,disp_true,mask):

    if len(disp_true[mask]) == 0:
        loss = 0.0
    else:
        loss = torch.sum(torch.abs(pred[mask]-disp_true[mask]))

    return loss