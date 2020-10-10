import numpy as np
import torch
import torch.nn as nn


class PairwiseLoss(nn.Module):
    def __init__(self):
        super(PairwiseLoss, self).__init__()

    def forward(self, x, y):
        diff = x - y
        return torch.sum(diff * diff)


class ESMTripletLoss(nn.Module):
    def __init__(self, device, m=0.01):
        super(ESMTripletLoss, self).__init__()
        self.m = m
        self.device = device

    def forward(self, anchor, positive, negative):
        batch_size = anchor.shape[0]
        diff_close = anchor - positive
        diff_depart = anchor - negative
        loss_close = torch.sum(diff_close * diff_close, axis=1)
        loss_depart = torch.sum(diff_depart * diff_depart, axis=1)
        factor = 1 - loss_depart / (loss_close + self.m)
        zeros = torch.zeros(batch_size).to(self.device)
        loss = torch.max(zeros, factor)
        return loss


if __name__ == '__main__':
    x = np.random.rand(1, 256)
    y = np.random.rand(1, 256)
    z = np.random.rand(1, 256)
    x = torch.Tensor(x)
    y = torch.Tensor(y)
    z = torch.Tensor(z)

    # pairwise_loss = PairwiseLoss()
    # loss = pairwise_loss(x, y)

    triplet_loss = ESMTripletLoss()
    loss = triplet_loss(x, y, z)