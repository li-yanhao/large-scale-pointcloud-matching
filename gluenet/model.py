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

        self.feat_nn = nn.Sequential(nn.Linear(out_dim, out_dim), nn.ReLU())
        self.gate_nn = nn.Sequential(nn.Linear(out_dim, 1), nn.ReLU())
        self.global_attention_pooling = GlobalAttentionPooling(gate_nn=self.gate_nn, feat_nn=self.feat_nn)


        for i in range(len(emb_dims)):
            self.conv.append(EdgeConv(
                emb_dims[i - 1] if i > 0 else in_dim,
                emb_dims[i],
                batch_norm=True)
            )
        self.conv.append(EdgeConv(
            emb_dims[-1],
            out_dim,
            batch_norm=True)
        )

    def forward(self, x, dev):
        batch_size, n_points, in_dim = x.shape
        h = x
        hs = []
        for i in range(len(self.conv)):
            g = self.knng(h).to(dev)
            h = h.view(batch_size * n_points, -1)
            h = self.conv[i](g, h)
            h = F.leaky_relu(h, 0.2)
            h = h.view(batch_size, n_points, -1)
            hs.append(h)
        g = self.knng(h).to(dev)
        h = h.view(n_points, -1)
        print(h.shape)
        y = self.global_attention_pooling(g, h)
        return y

x_raw = np.random.randn(1,100,3)
x = torch.Tensor(x_raw)

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = DescripNet(k=10, in_dim=3, emb_dims=[64, 128], out_dim=64)
opt = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)

model = model.to(dev)
x = x.to(dev)

model.train()
y = model(x, dev)


def main():
    pass

