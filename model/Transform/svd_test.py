import torch
import numpy as np
from scipy.spatial.transform import Rotation as R


def svd_test():
    N = 100
    rotation = R.from_rotvec((-np.pi / 1 + np.random.ranf() * 2 * np.pi / 1) * np.array([0, 0, 1])).as_matrix()
    translation = np.random.randn(3,1) * 20
    T_target_source = np.hstack([rotation, translation])
    T_target_source = np.vstack([T_target_source, np.array([0,0,0,1])])
    T_source_target = np.linalg.inv(T_target_source)

    print("T_target_source ground truth: \n", T_target_source)


    target_points = np.random.randn(N, 3)
    source_points = (T_source_target[:3,:3] @ target_points.transpose()).transpose() + T_source_target[:3,3]

    target_points = torch.Tensor(target_points)
    source_points = torch.Tensor(source_points)

    target_centers = target_points.mean(dim=0)
    source_centers = source_points.mean(dim=0)

    target_points_centered = target_points - target_centers
    source_points_centered = source_points - source_centers

    cov = source_points_centered.transpose(0,1) @ target_points_centered
    u, s, v = torch.svd(cov, some=False, compute_uv=True)

    v_neg = v.clone()
    v_neg[:, 2] *= -1
    rot_mat_neg = v_neg @ u.transpose(0,1)
    rot_mat_pos = v @ u.transpose(0,1)

    rot_mat = rot_mat_pos if torch.det(rot_mat_pos) > 0 else rot_mat_neg
    trans = -rot_mat @ source_centers + target_centers

    rot_mat = np.array(rot_mat)
    trans = np.array(trans).reshape(-1,1)
    T_target_source_restored = np.hstack([rot_mat, trans])
    T_target_source_restored = np.vstack([T_target_source_restored, np.array([0, 0, 0, 1])])

    print('T_target_source_restored:\n', T_target_source_restored)



if __name__ == '__main__':
    svd_test()