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
import os
import visdom

DATA_DIR = '/media/admini/My_data/0629'

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
            h = self.dropouts[i](h)

        h = self.proj_output(h)
        return h


MODEL_UNIT_TEST = False
if MODEL_UNIT_TEST:
    # x_raw = np.random.randn(1, 5000, 3)
    # x = torch.Tensor(x_raw)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model = DescripNet(k=10, in_dim=6, emb_dims=[32, 64, 64, 512], out_dim=64)
    model = DgcnnModel(k=10, feature_dims=[64, 128, 512], emb_dims=[256, 256], output_classes=256)
    opt = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    model = model.to(dev)

    # x = x.to(dev)
    model.train()

    Y = []
    X = []
    for i in range(270):
        x_raw = np.random.randn(1, 200, 3)
        x = torch.Tensor(x_raw).to(dev)
        y = model(x)
        X.append(x)
        Y.append(y)
    print("Done")

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

def compute_metrics(matches0, matches1, match_matrix_ground_truth):
    matches0 = np.array(matches0.cpu()).reshape(-1).squeeze() # M
    matches1 = np.array(matches1.cpu()).reshape(-1).squeeze() # N
    match_matrix_ground_truth = np.array(match_matrix_ground_truth.cpu()).squeeze()  # M*N

    matches0_idx_tuple = (np.arange(len(matches0)), matches0)
    matches1_idx_tuple = (np.arange(len(matches1)), matches1)

    matches0_precision_idx_tuple = (np.arange(len(matches0))[matches0>0], matches0[matches0>0])
    matches1_precision_idx_tuple = (np.arange(len(matches1))[matches1>0], matches1[matches1>0])

    matches0_recall_idx_tuple = (np.arange(len(matches0))[match_matrix_ground_truth[:-1, -1]==0], matches0[match_matrix_ground_truth[:-1, -1]==0])
    matches1_recall_idx_tuple = (np.arange(len(matches1))[match_matrix_ground_truth[-1, :-1]==0], matches1[match_matrix_ground_truth[-1, :-1]==0])

    match_0_acc = match_matrix_ground_truth[:-1, :][matches0_precision_idx_tuple].mean()
    match_1_acc = match_matrix_ground_truth.T[:-1, :][matches1_precision_idx_tuple].mean()

    metrics = {
        "matches0_acc": match_matrix_ground_truth[:-1, :][matches0_idx_tuple].mean(),
        "matches1_acc": match_matrix_ground_truth.T[:-1, :][matches1_idx_tuple].mean(),
        "matches0_precision": match_matrix_ground_truth[:-1, :][matches0_precision_idx_tuple].mean(),
        "matches1_precision": match_matrix_ground_truth.T[:-1, :][matches1_precision_idx_tuple].mean(),
        "matches0_recall": match_matrix_ground_truth[:-1, :][matches0_recall_idx_tuple].mean(),
        "matches1_recall": match_matrix_ground_truth.T[:-1, :][matches1_recall_idx_tuple].mean()
    }
    return metrics



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


RUN_PIPELINE = True
if RUN_PIPELINE:
    h5_filename = os.path.join(DATA_DIR, "submap_segments_downsampled.h5")
    correspondences_filename = os.path.join(DATA_DIR, "correspondences.json")
    gluenet_dataset = GlueNetDataset(h5_filename, correspondences_filename, mode='train')

    train_loader = DataLoader(gluenet_dataset, batch_size=1, shuffle=True)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    descriptor_dim = 128
    # model = DescripNet(k=10, in_dim=3, emb_dims=[64, 128, 128, 512], out_dim=descriptor_dim) # TODO: debug here
    model = DgcnnModel(k=5, feature_dims=[64, 128, 256], emb_dims=[256, 128], output_classes=descriptor_dim)
    model = model.to(dev)
    model.load_state_dict(torch.load(os.path.join(DATA_DIR, "model-dgcnn.pth"), map_location=dev))


    super_glue_config = {
        'descriptor_dim': descriptor_dim,
        'weights': '',
        'keypoint_encoder': [32, 64, 128],
        'GNN_layers': ['self', 'cross'] * 6,
        'sinkhorn_iterations': 150,
        'match_threshold': 0.02,
    }
    superglue = SuperGlue(super_glue_config)
    superglue = superglue.to(dev)
    superglue.load_state_dict(torch.load(os.path.join(DATA_DIR, "superglue-dgcnn.pth"), map_location=dev))

    opt = optim.Adam(list(model.parameters()) + list(superglue.parameters()), lr=1e-4, weight_decay=1e-6)
    num_epochs = 5
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, num_epochs, eta_min=0.001)

    scheduler.step()
    model.train()

    viz = visdom.Visdom()
    win_loss = viz.scatter(X=np.asarray([[0, 0]]))
    win_precision = viz.scatter(X=np.asarray([[0, 0]]))
    win_recall = viz.scatter(X=np.asarray([[0, 0]]))

    with tqdm(train_loader) as tq:
        item_idx = 0
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
                # descriptors_A.append(model(segment.to(dev), dev))
                descriptors_A.append(model(segment.to(dev)))
            for segment in segments_B:
                # descriptors_B.append(model(segment.to(dev), dev))
                descriptors_B.append(model(segment.to(dev)))
            descriptors_A = torch.cat(descriptors_A, dim=0).transpose(0, 1).reshape(1, descriptor_dim, -1)
            descriptors_B = torch.cat(descriptors_B, dim=0).transpose(0, 1).reshape(1, descriptor_dim, -1)
            data = {
                'descriptors0': descriptors_A,
                'descriptors1': descriptors_B,
                'keypoints0': centers_A.to(dev),
                'keypoints1': centers_B.to(dev),
            }

            match_output = superglue(data)


            loss = -match_output['scores'] * match_mask_ground_truth.to(dev)
            loss = loss.sum()

            loss.backward()
            opt.step()

            print("loss: {}".format(loss))

            # TODO: evaluate accuracy
            metrics = compute_metrics(match_output['matches0'], match_output['matches1'], match_mask_ground_truth)
            print("accuracies: matches0({}), matches1({})".format(metrics['matches0_acc'], metrics['matches1_acc']))
            print("precisions: matches0({}), matches1({})".format(metrics['matches0_precision'], metrics['matches1_precision']))
            print("recalls: matches0({}), matches1({})".format(metrics['matches0_recall'], metrics['matches1_recall']))

            viz.scatter(X=np.array([[item_idx, float(loss)]]),
                        name="train-loss",
                        win=win_loss,
                        update="append")
            viz.scatter(X=np.array([[item_idx, float(metrics['matches0_precision'])]]),
                        name="train-precision",
                        win=win_precision,
                        update="append")
            viz.scatter(X=np.array([[item_idx, float(metrics['matches0_recall'])]]),
                        name="train-recall",
                        win=win_recall,
                        update="append")

            item_idx += 1
            if item_idx % 200 == 0:
                # TODO: save weight file
                torch.save(model.state_dict(), os.path.join(DATA_DIR, "model-dgcnn.pth"))
                torch.save(superglue.state_dict(), os.path.join(DATA_DIR, "superglue-dgcnn.pth"))
                print("model weights saved in {}".format(DATA_DIR))

            # TODO: draw a curve to supervise
            # TODO: increase the complexity of descriptor learning model


def main():
    pass
