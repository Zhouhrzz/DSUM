from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
from utils import *
import torch.nn.functional as F
import os
#import tqdm
import scipy.io as scio
# Training settings
from model import AlexNet
import dataset

def test_pre(model, t_loader, Instances=False):

    v_correct, v_sum = 0, 0
    start = 0
    for ind2, (xv, yv) in enumerate(t_loader):
        xv, yv = xv.cuda(), yv.cuda()
        N, V, C, H, W = xv.size()
        xv = Variable(xv).view(-1, C, H, W).cuda()

        v_feature, _, v_pred, _, _ = model.forward(xv, bs=N, Instance=Instances)

        v_pred_label = torch.max(v_pred, 1)[1]
        v_equal = torch.eq(v_pred_label, yv).float()
        v_correct += torch.sum(v_equal).item()
        v_sum += len(yv)
        if start == 0:
            all_feature = v_feature.detach().cpu()
            label = yv.view(-1,1).detach().cpu()
            pred_label = v_pred_label.view(-1, 1).detach().cpu()
            start += 1
        else:
            all_feature = torch.cat((all_feature, v_feature.detach().cpu()), dim=0)
            label = torch.cat((label, yv.view(-1,1).detach().cpu()), dim=0)
            pred_label = torch.cat((pred_label, v_pred_label.view(-1, 1).detach().cpu()), dim=0)

    v_acc = v_correct / v_sum

    return all_feature, label, pred_label, v_acc