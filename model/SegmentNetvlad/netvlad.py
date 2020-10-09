import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
import numpy as np

# based on https://github.com/lyakaap/NetVLAD-pytorch/blob/master/netvlad.py
# class NetVLAD(nn.Module):
#     """NetVLAD layer implementation"""
#
#     def __init__(self, num_clusters=64, dim=128,
#                  normalize_input=True, vladv2=False):
#         """
#         Args:
#             num_clusters : int
#                 The number of clusters
#             dim : int
#                 Dimension of descriptors
#             alpha : float
#                 Parameter of initialization. Larger value is harder assignment.
#             normalize_input : bool
#                 If true, descriptor-wise L2 normalization is applied to input.
#             vladv2 : bool
#                 If true, use vladv2 otherwise use vladv1
#         """
#         super(NetVLAD, self).__init__()
#         self.num_clusters = num_clusters
#         self.dim = dim
#         self.alpha = 0
#         self.vladv2 = vladv2
#         self.normalize_input = normalize_input
#         self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=vladv2)
#         self.centroids = nn.Parameter(torch.rand(num_clusters, dim))
#
#     def init_params(self, clsts, traindescs):
#         #TODO replace numpy ops with pytorch ops
#         if self.vladv2 == False:
#             clstsAssign = clsts / np.linalg.norm(clsts, axis=1, keepdims=True)
#             dots = np.dot(clstsAssign, traindescs.T)
#             dots.sort(0)
#             dots = dots[::-1, :] # sort, descending
#
#             self.alpha = (-np.log(0.01) / np.mean(dots[0,:] - dots[1,:])).item()
#             self.centroids = nn.Parameter(torch.from_numpy(clsts))
#             self.conv.weight = nn.Parameter(torch.from_numpy(self.alpha*clstsAssign).unsqueeze(2).unsqueeze(3))
#             self.conv.bias = None
#         else:
#             knn = NearestNeighbors(n_jobs=-1) #TODO faiss?
#             knn.fit(traindescs)
#             del traindescs
#             dsSq = np.square(knn.kneighbors(clsts, 2)[1])
#             del knn
#             self.alpha = (-np.log(0.01) / np.mean(dsSq[:,1] - dsSq[:,0])).item()
#             self.centroids = nn.Parameter(torch.from_numpy(clsts))
#             del clsts, dsSq
#
#             self.conv.weight = nn.Parameter(
#                 (2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1)
#             )
#             self.conv.bias = nn.Parameter(
#                 - self.alpha * self.centroids.norm(dim=1)
#             )
#
#     def forward(self, x):
#         N, C = x.shape[:2]
#
#         if self.normalize_input:
#             x = F.normalize(x, p=2, dim=1)  # across descriptor dim
#
#         # soft-assignment
#         soft_assign = self.conv(x).view(N, self.num_clusters, -1)
#         soft_assign = F.softmax(soft_assign, dim=1)
#
#         x_flatten = x.view(N, C, -1)
#
#         # calculate residuals to each clusters
#         vlad = torch.zeros([N, self.num_clusters, C], dtype=x.dtype, layout=x.layout, device=x.device)
#         for C in range(self.num_clusters): # slower than non-looped, but lower memory usage
#             residual = x_flatten.unsqueeze(0).permute(1, 0, 2, 3) - \
#                     self.centroids[C:C+1, :].expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
#             residual *= soft_assign[:,C:C+1,:].unsqueeze(2)
#             vlad[:,C:C+1,:] = residual.sum(dim=-1)
#
#         vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
#         vlad = vlad.view(x.size(0), -1)  # flatten
#         vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize
#
#         return vlad


class NetVLAD(nn.Module):
    """NetVLAD layer implementation"""

    def __init__(self, num_clusters=64, dim=128, alpha=100.0, outdim=10,
                 normalize_input=True):
        """
        Args:
            num_clusters : int
                The number of clusters
            dim : int
                Dimension of descriptors
            alpha : float
                Parameter of initialization. Larger value is harder assignment.
            normalize_input : bool
                If true, descriptor-wise L2 normalization is applied to input.
        """
        super(NetVLAD, self).__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.alpha = alpha
        self.normalize_input = normalize_input
        self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=True)
        self.linear = nn.Linear(dim, num_clusters, bias=True)
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim))
        self._init_params()

    def _init_params(self):
        self.conv.weight = nn.Parameter(
            (2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1)
        )
        self.conv.bias = nn.Parameter(
            - self.alpha * self.centroids.norm(dim=1)
        )

    def forward(self, x):
        N, D = x.shape[:2]

        if self.normalize_input:
            x = F.normalize(x, p=2, dim=0)  # across descriptor dim

        # soft-assignment
        soft_assign = self.linear(x) # N, num_clusters,
        # soft_assign = self.conv(x).view(N, self.num_clusters, -1)
        soft_assign = F.softmax(soft_assign, dim=1)

        x_flatten = x.view(N, D, 1)
        soft_assign.unsqueeze(1)
        residual = x.unsqueeze(1) - self.centroids.unsqueeze(0) # N, D, num_clusters

        vlad = torch.einsum('nok,nkd->okd', soft_assign.unsqueeze(1), residual)
        vlad = F.normalize(vlad, p=2, dim=1)

        # torch.einsum('bhnm,bdhm->bdhn', prob, value)
        #
        # # calculate residuals to each clusters
        # residual = x_flatten.expand(self.num_clusters, -1, -1, -1).permute(1, 0, 2, 3) - \
        #            self.centroids.expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
        # residual *= soft_assign.unsqueeze(2)
        # vlad = residual.sum(dim=-1)
        #
        # vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        # vlad = vlad.view(x.size(0), -1)  # flatten
        # vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize

        return vlad


class EmbedNet(nn.Module):
    def __init__(self, base_model, net_vlad, inter_dim=128):
        super(EmbedNet, self).__init__()
        self.base_model = base_model
        self.net_vlad = net_vlad

    def forward(self, x):
        x = self.base_model(x)
        embedded_x = self.net_vlad(x) # 9 * 16384
        return embedded_x


class TripletNet(nn.Module):
    def __init__(self, embed_net):
        super(TripletNet, self).__init__()
        self.embed_net = embed_net

    def forward(self, a, p, n):
        embedded_a = self.embed_net(a)
        embedded_p = self.embed_net(p)
        embedded_n = self.embed_net(n)
        return embedded_a, embedded_p, embedded_n

    def feature_extract(self, x):
        return self.embed_net(x)


# class ContrastiveLoss(torch.nn.Module):
#     """
#     Contrastive loss function.
#     Based on:
#     """
#
#     def __init__(self, margin=1.0):
#         super(ContrastiveLoss, self).__init__()
#         self.margin = margin
#
#     def check_type_forward(self, in_types):
#         assert len(in_types) == 3
#
#         x0_type, x1_type, y_type = in_types
#         assert x0_type.size() == x1_type.shape
#         assert x1_type.size()[0] == y_type.shape[0]
#         assert x1_type.size()[0] > 0
#         assert x0_type.dim() == 2
#         assert x1_type.dim() == 2
#         assert y_type.dim() == 1
#
#     def forward(self, x0, x1, y):
#         self.check_type_forward((x0, x1, y))
#
#         # euclidian distance
#         diff = x0 - x1
#         dist_sq = torch.sum(torch.pow(diff, 2), 1)
#         dist = torch.sqrt(dist_sq)
#
#         mdist = self.margin - dist
#         dist = torch.clamp(mdist, min=0.0)
#         loss = y * dist_sq + (1 - y) * torch.pow(dist, 2)
#         loss = torch.sum(loss) / 2.0 / x0.size()[0]
#         return loss

class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, is_negative):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1 - is_negative) * torch.pow(euclidean_distance, 2) +
                                      (is_negative) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive