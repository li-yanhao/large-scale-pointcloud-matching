#!/usr/bin/env python3

# Author: Yanhao Li

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def torch_icp(target_points, source_points, correspondences):
    B, M, _ = target_points.shape
    _, N, _ = source_points.shape
    assert (B == 1)
    # normalize
    weights_normalized = correspondences[:, :M, :N] / correspondences[:, :M, :N].sum(dim=1)
    target_points_predicted = torch.einsum('bmn,bmd->bnd', weights_normalized, target_points)

    # target_points_predicted = torch.einsum('bmn,bmd->bnd', correspondences[:, :M, :N], target_points)

    target_predicted_centers = target_points_predicted.mean(dim=1)
    source_centers = source_points.mean(dim=1)

    target_points_predicted_centered = target_points_predicted - target_predicted_centers  # B*N*3
    source_points_centered = source_points - source_centers  # B*N*3
    # cov = torch.einsum('bjn,bnk->bjk', target_points_centered.permute(0,2,1), source_points_centered)
    target_points_predicted_centered_weighted = torch.einsum('ijk,ij->ijk',target_points_predicted_centered, correspondences[:, :M, :N].sum(dim=1))
    cov = source_points_centered.permute(0, 2, 1) @ target_points_predicted_centered_weighted
    u, s, v = torch.svd(cov, some=False, compute_uv=True)

    v_neg = v.clone()
    v_neg[:, 2] *= -1
    rot_mat_neg = v_neg @ u.transpose(-2,-1)
    rot_mat_pos = v @ u.transpose(-2,-1)

    R_target_source = torch.where(torch.det(rot_mat_pos)[:, None, None] > 0, rot_mat_pos, rot_mat_neg)
    t_target_source = torch.squeeze(-R_target_source @ source_centers[...,None], -1) + target_predicted_centers


    return R_target_source, t_target_source


class RTNet(nn.Module):
    # Rigid transform module to compute differentiable rigid transform given points and correspondences
    def __init__(self):
        super(RTNet, self).__init__()

    # target_points: B*M*3
    # source_points: B*N*3
    # correspondences: B*(M+1)*(N+1)
    def forward(self, target_points, source_points, correspondences):
        B, M, _ = target_points.shape
        _, N, _ = source_points.shape
        assert(B == 1)
        correspondences[:,:M,:N]
        weights_normalized = correspondences[:,:M,:N] / correspondences[:,:M,:N].sum(dim=1)
        target_points_predicted = torch.einsum('bmn,bmd->bnd', weights_normalized, target_points)

        target_centers = target_points.mean(dim=1)
        source_centers = source_points.mean(dim=1)

        target_points_centered = target_points - target_centers # B*N*3
        source_points_centered = source_points - source_centers # B*N*3
        # cov = torch.einsum('bjn,bnk->bjk', target_points_centered.permute(0,2,1), source_points_centered)
        cov = source_points_centered.permute(0,2,1) @ target_points_centered
        u, s, v = torch.svd(cov, some=False, compute_uv=True)

        v_neg = v.clone()
        v_neg[:, 2] *= -1
        rot_mat_neg = v_neg @ u.transpose(0, 1)
        rot_mat_pos = v @ u.transpose(0, 1)

        R_target_source = torch.where(torch.det(rot_mat_pos)[:, None, None] > 0, rot_mat_pos, rot_mat_neg)
        t_target_source = -R_target_source @ source_centers + target_centers

        return R_target_source, t_target_source


        # weights_normalized = weights[..., None] / (torch.sum(weights[..., None], dim=1, keepdim=True) + _EPS)
        # centroid_a = torch.sum(a * weights_normalized, dim=1)
        # centroid_b = torch.sum(b * weights_normalized, dim=1)
        # a_centered = a - centroid_a[:, None, :]
        # b_centered = b - centroid_b[:, None, :]
        # cov = a_centered.transpose(-2, -1) @ (b_centered * weights_normalized)
        #
        # # Compute rotation using Kabsch algorithm. Will compute two copies with +/-V[:,:3]
        # # and choose based on determinant to avoid flips
        # u, s, v = torch.svd(cov, some=False, compute_uv=True)
        #
        #
        # rot_mat_pos = v @ u.transpose(-1, -2)
        # v_neg = v.clone()
        # v_neg[:, :, 2] *= -1
        # rot_mat_neg = v_neg @ u.transpose(-1, -2)
        # rot_mat = torch.where(torch.det(rot_mat_pos)[:, None, None] > 0, rot_mat_pos, rot_mat_neg)
