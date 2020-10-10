#!/usr/bin/env python3

# Author: Yanhao Li

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dgl.nn.pytorch import KNNGraph, EdgeConv

from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import open3d as o3d
from model.Descriptor.descriptor_dataset import *


class DescNet(nn.Module):
    def __init__(self, dgcnn, meta_info_net):
        super(DescNet, self).__init__()
        self.dgcnn = dgcnn
        self.meta_info_net = meta_info_net

    def forward(self, segment_info):
        segment_desc = self.dgcnn(segment_info['segment'])
        segment_meta_embedding = self.meta_info_net(segment_info['segment_center'], segment_info['segment_scale'])


class MetaInfoModel(nn.Module):
    def __init__(self, dims=[3,64]):
        super(MetaInfoModel, self).__init__()
        self.nn_list = nn.ModuleList()
        assert len(dims) > 1
        for i in range(len(dims)-1):
            self.nn_list.append(nn.Linear(dims[i], dims[i+1]))

    def forward(self, x):
        for layer in self.nn_list:
            x = layer(x)
            x = F.relu(x, True)
        return x


class DgcnnModel(nn.Module):
    def __init__(self, k, feature_dims, emb_dims, output_classes, input_dims=3,
                 dropout_prob=0.5):
        super(DgcnnModel, self).__init__()

        self.nng = KNNGraph(k)
        self.conv = nn.ModuleList()

        self.num_layers = len(feature_dims)
        for i in range(self.num_layers):
            self.conv.append(EdgeConv(
                feature_dims[i - 1] if i > 0 else input_dims,
                feature_dims[i],
                batch_norm=True))

        self.proj = nn.Linear(sum(feature_dims), emb_dims[0])

        self.embs = nn.ModuleList()
        self.bn_embs = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        self.num_embs = len(emb_dims) - 1
        for i in range(1, self.num_embs + 1):
            self.embs.append(nn.Linear(
                # * 2 because of concatenation of max- and mean-pooling
                emb_dims[i - 1] if i > 1 else (emb_dims[i - 1] * 2),
                emb_dims[i]))
            # self.bn_embs.append(nn.BatchNorm1d(emb_dims[i]))
            self.dropouts.append(nn.Dropout(dropout_prob))

        self.proj_output = nn.Linear(emb_dims[-1], output_classes)

    # 80 * 3 batches * 200 points => 7GB GPU memory
    def forward(self, x):
        hs = []
        batch_size, n_points, x_dims = x.shape
        h = x

        for i in range(self.num_layers):
            g = self.nng(h).to(h.device)
            h = h.view(batch_size * n_points, -1)
            h = self.conv[i](g, h)
            h = F.leaky_relu(h, 0.2)
            h = h.view(batch_size, n_points, -1)
            hs.append(h)

        h = torch.cat(hs, 2)
        h = self.proj(h)
        h_max, _ = torch.max(h, 1)
        h_avg = torch.mean(h, 1)
        h = torch.cat([h_max, h_avg], 1)

        for i in range(self.num_embs):
            h = self.embs[i](h)
            # h = self.bn_embs[i](h)
            h = F.leaky_relu(h, 0.2)
            # h = self.dropouts[i](h)

        h = self.proj_output(h)
        return h


if __name__ == "__main__":
    # h5_filename = "/media/admini/My_data/submap_database/00/submap_segments.h5"
    # correspondences_filename = "/media/admini/My_data/submap_database/00/correspondences.txt"
    #
    # descriptor_dataset = DescriptorDataset(h5_filename, correspondences_filename, mode='train')
    # train_loader = DataLoader(descriptor_dataset, batch_size=1, shuffle=True)
    # n = 0
    # for item in train_loader:
    #     anchor, positive, negative = item

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DgcnnModel(k=10, feature_dims=[64, 128, 512], emb_dims=[256, 256], output_classes=256)
    opt = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    model = model.to(dev)

    # x = x.to(dev)
    model.train()


    anchor = {
        "segment" : np.random.randn(1, 256, 3),
        "segment_scale": np.random.randn(3),
        # "segment_center": np.random.randn(3)
    }

    positive = {
        "segment": np.random.randn(1, 256, 3),
        "segment_scale": np.random.randn(3),
        # "segment_center": np.random.randn(3)
    }

    negative = {
        "segment": np.random.randn(1, 256, 3),
        "segment_scale": np.random.randn(3),
        # "segment_center": np.random.randn(3)
    }

    anchor_desc = model(torch.Tensor(anchor['segment']).to(dev))
    positive_desc = model(torch.Tensor(positive['segment']).to(dev))
    negative_desc = model(torch.Tensor(negative['segment']).to(dev))
    criterion = nn.TripletMarginLoss(margin=0.5, p=2, reduction='sum')
    loss = criterion(anchor_desc, positive_desc, negative_desc)
    loss.backward()

    # MODEL_DIR = '/media/admini/My_data/0629'
    # dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # descriptor_dim = 256
    # model = DescNet(k=5, feature_dims=[64, 128, 256], emb_dims=[512, 256], output_classes=descriptor_dim)
    # model.load_state_dict(
    #     torch.load(os.path.join(MODEL_DIR, "model-dgcnn-no-dropout.pth"), map_location=torch.device('cpu')))
    #
    # model.train()
    # model = model.to(dev)
    #
    # descriptors = []
    # with torch.no_grad():
    #     # segments_A = [segment.to(dev) for segment in segments_A]
    #     # segments_B = [segment.to(dev) for segment in segments_B]
    #     # descriptors_A = torch.Tensor.new_empty(1, 256, len(segments_A), device=dev)
    #     # descriptors_B = torch.Tensor.new_empty(1, 256, len(segments_B), device=dev)
    #
    #     descriptors_B = []
    #     # for i in range(len(segments_A)):
    #     #     descriptors_A[0, :, i] = model(segments_A[i], dev)
    #     # for i in range(len(segments_B)):
    #     #     descriptors_B.append(model(segment, dev))
    #     for segment in segments:
    #         # descriptors_A.append(model(segment.to(dev), dev))
    #         descriptors.append(model(torch.Tensor(segment).reshape(1, -1, 3).to(dev)))
    #
    # descriptors = np.array([np.array(descriptor.cpu()).reshape(-1) for descriptor in descriptors])
    # np.save("descriptors_database.npy", descriptors)
    #
    # print(descriptors)
    #
    # # save descriptors somewhere
