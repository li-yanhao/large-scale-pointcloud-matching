#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dgl.nn.pytorch import KNNGraph, EdgeConv
from dgl.nn.pytorch.glob import GlobalAttentionPooling
import numpy as np



# DescripNet
# input: points(Na, 3)
# embedding: DGCNN -> DGCNN -> GlobalAttentionPooling
# output: descriptor(d)
class DescripNet(nn.Module):
    def __init__(self, k, in_dim : int, emb_dims : list, out_dim : int):
        super(DescripNet, self).__init__()

        self.knng = KNNGraph(k)
        self.conv = nn.ModuleList()

        self.feat_nn = nn.Sequential(nn.Linear(emb_dims[-2], emb_dims[-1]), nn.ReLU())
        self.gate_nn = nn.Sequential(nn.Linear(emb_dims[-2], 1), nn.ReLU())
        self.global_attention_pooling = GlobalAttentionPooling(gate_nn=self.gate_nn, feat_nn=self.feat_nn)
        self.last_layer = nn.Linear(emb_dims[-1], out_dim)
        for i in range(len(emb_dims)-1):
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
        for i in range(len(self.conv)-1):
            g = self.knng(h).to(dev)
            h = h.view(batch_size * n_points, -1)
            h = self.conv[i](g, h)
            h = F.leaky_relu(h, 0.2)
            h = h.view(batch_size, n_points, -1)
            # hs.append(h)
        print(h.shape)
        g = self.knng(h).to(dev)
        h = h.view(batch_size * n_points, -1)
        h = self.global_attention_pooling(g, h)
        y = self.last_layer(h)
        return y

x_raw = np.random.randn(1,5000,3)
x = torch.Tensor(x_raw)

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = DescripNet(k=10, in_dim=6, emb_dims=[32,64,64,512], out_dim=64)
opt = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
model = model.to(dev)

x = x.to(dev)
model.train()

# x_raw = np.random.randn(1,5000,3)
# x = torch.Tensor(x_raw)
# y = model(x, dev)

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

def main():
    pass

