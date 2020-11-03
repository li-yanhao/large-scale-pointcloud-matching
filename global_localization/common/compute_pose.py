import torch
import numpy as np


def compute_relative_pose(target_points, source_points):
    """
    :param target_keypoints: N * 2
    :param source_keypoints: N * 2
    :return: T_target_source_best: 4 * 4
             score: float
    """
    assert(len(target_points) == len(target_points))

    target_points = torch.Tensor(target_points)
    source_points = torch.Tensor(source_points)

    target_centers = target_points.mean(dim=0)
    source_centers = source_points.mean(dim=0)

    target_points_centered = target_points - target_centers
    source_points_centered = source_points - source_centers

    cov = source_points_centered.transpose(0, 1) @ target_points_centered
    u, s, v = torch.svd(cov, some=True, compute_uv=True)

    v_neg = v.clone()
    v_neg[:, 1] *= -1
    rot_mat_neg = v_neg @ u.transpose(0, 1)
    rot_mat_pos = v @ u.transpose(0, 1)

    rot_mat = rot_mat_pos if torch.det(rot_mat_pos) > 0 else rot_mat_neg
    trans = -rot_mat @ source_centers + target_centers

    rot_mat = np.array(rot_mat)
    trans = np.array(trans).reshape(-1, 1)
    T_target_source_restored = np.hstack([rot_mat, trans])
    T_target_source_restored = np.vstack([T_target_source_restored, np.array([0, 0, 1])])

    # print('T_target_source_restored:\n', T_target_source_restored)
    return T_target_source_restored


def compute_relative_pose_with_ransac(target_keypoints, source_keypoints):
    """
    :param target_keypoints: N * 2
    :param source_keypoints: N * 2
    :return: T_target_source_best: 4 * 4
             score: float
    """
    assert(target_keypoints.shape == source_keypoints.shape)
    num_matches = len(target_keypoints)
    n, k = 1000, 10
    if num_matches < k:
        return None, None

    target_keypoints = torch.Tensor(target_keypoints)
    source_keypoints = torch.Tensor(source_keypoints)


    selections = np.random.choice(num_matches, (n, k), replace=True)

    target_sub_keypoints = target_keypoints[selections] # N * k * 2
    source_sub_keypoints = source_keypoints[selections] # N * k * 2
    target_centers = target_sub_keypoints.mean(dim=1) # N * 2
    source_centers = source_sub_keypoints.mean(dim=1) # N * 2
    target_sub_keypoints_centered = target_sub_keypoints - target_centers.unsqueeze(1)
    source_sub_keypoints_centered = source_sub_keypoints - source_centers.unsqueeze(1)
    cov = source_sub_keypoints_centered.transpose(1, 2) @ target_sub_keypoints_centered
    u, s, v = torch.svd(cov) # u: N*2*2, s: N*2, v: N*2*2

    v_neg = v.clone()
    v_neg[:,:, 1] *= -1

    rot_mats_neg = v_neg @ u.transpose(1, 2)
    rot_mats_pos = v @ u.transpose(1, 2)
    determinants = torch.det(rot_mats_pos)

    rot_mats_neg_list = [rot_mat_neg for rot_mat_neg in rot_mats_neg]
    rot_mats_pos_list = [rot_mat_neg for rot_mat_neg in rot_mats_pos]

    rot_mats_list = [rot_mat_pos if determinant > 0 else rot_mat_neg for (determinant, rot_mat_pos, rot_mat_neg) in zip(determinants, rot_mats_pos_list, rot_mats_neg_list)]
    rotations = torch.stack(rot_mats_list) # N * 2 * 2
    translations = torch.einsum("nab,nb->na", -rotations, source_centers) + target_centers # N * 2
    diff = source_keypoints @ rotations.transpose(1,2) + translations.unsqueeze(1) - target_keypoints
    distances_squared = torch.sum(diff * diff, dim=2)

    distance_tolerance = 1.0
    scores = (distances_squared < (distance_tolerance**2)).sum(dim=1)
    score = torch.max(scores)
    best_index = torch.argmax(scores)
    rotation = rotations[best_index]
    translation = translations[best_index]
    T_target_source = torch.cat((rotation, translation[...,None]), dim=1)
    T_target_source = torch.cat((T_target_source, torch.Tensor([[0,0,1]])), dim=0)
    return T_target_source, score
