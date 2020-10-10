import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dgl.nn.pytorch import KNNGraph, EdgeConv
from dgl.nn.pytorch.glob import GlobalAttentionPooling

from model.SapientNet.superglue import SuperGlue
from model.SapientNet.dataset import GlueNetDataset
from model.Descriptor.descnet import DescNet

from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import visdom


class MatchNet(nn.Module):
    def __init__(self, descnet, gluenet, desc_dim, device):
        self.descnet = descnet
        self.gluenet = gluenet
        self.desc_dim = desc_dim
        self.device = device

    def forward(self, segments_target, segments_source, centers_target, centers_source, scales_target, scales_source):
        descriptors_target = []
        descriptors_source = []
        for segment in segments_target:
            descriptors_target.append(self.descnet(segment.to(self.device)))
        for segment in segments_source:
            descriptors_source.append(self.descnet(segment.to(self.device)))
        descriptors_target = torch.cat(descriptors_target, dim=0).transpose(0, 1).reshape(1, self.desc_dim, -1)
        descriptors_source = torch.cat(descriptors_source, dim=0).transpose(0, 1).reshape(1, self.desc_dim, -1)
        data = {
            'descriptors0': descriptors_target,
            'descriptors1': descriptors_source,
            'keypoints0': torch.cat([centers_target, scales_target], dim=1).to(self.device),
            'keypoints1': torch.cat([centers_source, scales_source], dim=1).to(self.device),
        }

        return self.gluenet(data)