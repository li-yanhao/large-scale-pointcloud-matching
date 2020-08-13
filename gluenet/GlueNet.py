#!/usr/bin/env python3

# Author: Yanhao Li

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dgl.nn.pytorch import KNNGraph, EdgeConv
from dgl.nn.pytorch.glob import GlobalAttentionPooling

from gluenet.superglue import SuperGlue
from gluenet.dataset import GlueNetDataset
from torch.utils.data import DataLoader
from tqdm import tqdm



# DescripNet
# input: points(Na, 3)
# embedding: DGCNN -> DGCNN -> GlobalAttentionPooling
# output: descriptor(d)
class DescripNet(nn.Module):
    def __init__(self, k, in_dim: int, emb_dims: list, out_dim: int):
        super(DescripNet, self).__init__()

        self.knng = KNNGraph(k)
        self.conv = nn.ModuleList()

        self.feat_nn = nn.Sequential(nn.Linear(emb_dims[-2], emb_dims[-1]), nn.ReLU())
        self.gate_nn = nn.Sequential(nn.Linear(emb_dims[-2], 1), nn.ReLU())
        self.global_attention_pooling = GlobalAttentionPooling(gate_nn=self.gate_nn, feat_nn=self.feat_nn)
        self.last_layer = nn.Linear(emb_dims[-1], out_dim)
        for i in range(len(emb_dims) - 1):
            self.conv.append(EdgeConv(
                emb_dims[i - 1] if i > 0 else in_dim,
                emb_dims[i],
                batch_norm=True)
            )
        # self.conv.append(self.global_attention_pooling)

    def forward(self, x, dev):
        batch_size, n_points, in_dim = x.shape
        h = x
        # hs = []
        for i in range(len(self.conv) - 1):
            g = self.knng(h).to(dev)
            h = h.view(batch_size * n_points, -1)
            h = self.conv[i](g, h)
            h = F.leaky_relu(h, 0.2)
            h = h.view(batch_size, n_points, -1)
            # hs.append(h)
        # print(h.shape)
        g = self.knng(h).to(dev)
        h = h.view(batch_size * n_points, -1)
        h = self.global_attention_pooling(g, h)
        y = self.last_layer(h)
        return y

if False:
    x_raw = np.random.randn(1, 5000, 3)
    x = torch.Tensor(x_raw)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DescripNet(k=10, in_dim=6, emb_dims=[32, 64, 64, 512], out_dim=64)
    opt = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    model = model.to(dev)

    x = x.to(dev)
    model.train()

    Y = []
    X = []
    for i in range(1200):
        x_raw = np.random.randn(1, 200, 6)
        x = torch.Tensor(x_raw)
        x = x.to(dev)
        y = model(x, dev)
        X.append(x)
        Y.append(y)


# model => 400 MB
# 6695 points => 176 MB GPU memory
# 6453 points => 349 MB
# 4997 points =>
# 3341 points => 10 MB
# 500pts * 237batches = 118500 => out of memory
# 200pts * 605batches = 120000 => out of memory
# 2000pts * 55batches = 110000 => out of memory

# downsample: 200000 * (x,y,z) => 8000*(x,y,z,alpha,beta,gamma)

# 200*6 * 369batches => out of memory


if False:
    super_glue_config = {
        'descriptor_dim': 256,
        'weights': '',
        'keypoint_encoder': [32, 64, 128, 256],
        'GNN_layers': ['self', 'cross'] * 9,
        'sinkhorn_iterations': 100,
        'match_threshold': 0.2,
    }

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    superglue = SuperGlue(super_glue_config)

    n0 = 500
    n1 = 400
    d = 256
    W = 800
    H = 600
    L = 3
    x_for_superglue = {
        'descriptors0': torch.Tensor(np.random.randn(1, d, n0)).to(dev), # 1 * d * n
        'keypoints0': torch.Tensor(np.random.randn(1, n0, 3)).to(dev), # 1 * n * 3
        'descriptors1': torch.Tensor(np.random.randn(1, d, n1)).to(dev), # 1 * d * n
        'keypoints1': torch.Tensor(np.random.randn(1, n1, 3)).to(dev), # 1 * n * 3
        'image0': torch.Tensor(np.random.randn(1, 1, H, W, L)).to(dev), # 1 * 1 * H * W
        'image1': torch.Tensor(np.random.randn(1, 1, H, W, L)).to(dev), # 1 * 1 * H * W
        'scores0': torch.Tensor(np.random.randn(1, n0)).to(dev), # 1 * n
        'scores1': torch.Tensor(np.random.randn(1, n1)).to(dev) # 1 * n
    }

# torch.Size([1, 988, 2])
# torch.Size([1, 988])
# torch.Size([1, 256, 988])
# torch.Size([1, 1, 480, 640])
# torch.Size([1, 1, 480, 640])
# torch.Size([1, 988, 2])
# torch.Size([1, 988])
# torch.Size([1, 256, 988])

    superglue.train().to(dev)
    y_for_superglue = superglue(x_for_superglue)


    print(y_for_superglue)
    for key in y_for_superglue:
        print(key, y_for_superglue[key].shape)
    # matches0 torch.Size([1, 500])
    # matches1 torch.Size([1, 400])
    # matching_scores0 torch.Size([1, 500])
    # matching_scores1 torch.Size([1, 400])

if True:
    h5_filename = "/home/li/Documents/submap_segments.h5"
    correspondences_filename = "/home/li/Documents/correspondences.json"
    gluenet_dataset = GlueNetDataset(h5_filename, correspondences_filename, mode='train')

    train_loader = DataLoader(gluenet_dataset, batch_size=1, shuffle=False)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DescripNet(k=10, in_dim=3, emb_dims=[32, 64, 64, 512], out_dim=256)
    model = model.to(dev)

    super_glue_config = {
        'descriptor_dim': 256,
        'weights': '',
        'keypoint_encoder': [32, 64, 128, 256],
        'GNN_layers': ['self', 'cross'] * 9,
        'sinkhorn_iterations': 100,
        'match_threshold': 0.2,
    }
    superglue = SuperGlue(super_glue_config)
    superglue = superglue.to(dev)

    opt = optim.Adam(model.parameters(), lr=0.1, weight_decay=1e-4)
    num_epochs = 5
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, num_epochs, eta_min=0.001)



    scheduler.step()
    model.train()
    with tqdm(train_loader) as tq:
        for centers_A, centers_B, segments_A, segments_B, match_mask_ground_truth in tq:
            # segments_A = [segment.to(dev) for segment in segments_A]
            # segments_B = [segment.to(dev) for segment in segments_B]
            # descriptors_A = torch.Tensor.new_empty(1, 256, len(segments_A), device=dev)
            # descriptors_B = torch.Tensor.new_empty(1, 256, len(segments_B), device=dev)
            descriptors_A = []
            descriptors_B = []
            opt.zero_grad()
            # for i in range(len(segments_A)):
            #     descriptors_A[0, :, i] = model(segments_A[i], dev)
            # for i in range(len(segments_B)):
            #     descriptors_B.append(model(segment, dev))
            for segment in segments_A:
                descriptors_A.append(model(segment.to(dev), dev))
            for segment in segments_B:
                descriptors_B.append(model(segment.to(dev), dev))
            descriptors_A = torch.cat(descriptors_A, dim=0).transpose(0, 1).reshape(1, 256, -1)
            descriptors_B = torch.cat(descriptors_B, dim=0).transpose(0, 1).reshape(1, 256, -1)
            data = {
                'descriptors0': descriptors_A,
                'descriptors1': descriptors_B,
                'keypoints0': centers_A.to(dev),
                'keypoints1': centers_B.to(dev),
            }

            superglue(data)
            print(data)

            loss = -data['score'].log() * match_mask_ground_truth
            opt.step()

def main():
    pass
