#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import KNNGraph, EdgeConv
from dgl.nn.pytorch.glob import GlobalAttentionPooling



# DescripNet
# input: points(Na, 3)
# embedding: DGCNN -> DGCNN -> GlobalAttentionPooling
# output: descriptor(d)
class DescripNet(nn.Module):
    def __init__(self, k, in_dim : int, emb_dims, out_dim : int):
        super(DescripNet, self).__init__()

        self.knng = KNNGraph(k)
        self.conv = nn.ModuleList()

        self.feat_nn = nn.Sequential(nn.Linear(emb_dims[-1], out_dim), F.relu())
        self.gat_nn = nn.Sequential(nn.Linear(emb_dims[-1], out_dim), F.relu())
        self.global_attention_pooling = GlobalAttentionPooling(gat_nn=self.gat_nn, feat_nn=self.feat_nn)

        self.num_layers = len(emb_dims)

        for i in range(self.num_layers):
            self.conv.append(EdgeConv(
                emb_dims[i - 1] if i > 0 else in_dim,
                emb_dims[i],
                batch_norm=True)
            )


    def forward(self, x):
        batch_size, n_points, in_dim = x.shape
        h = x
        hs = []
        for i in range(self.num_layers):
            g = self.knng(h)
            h = h.view(batch_size * n_points, -1)
            h = self.conv[i](g, h)
            h = F.leaky_relu(h, 0.2)
            h = h.view(batch_size, n_points, -1)
            hs.append(h)
        g = self.knng(h)
        y = self.global_attention_pooling(g, h)
        return y



def main():
    pass

