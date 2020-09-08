from scipy.spatial.transform import Rotation as R
import torch
import open3d as o3d
import numpy as np

def make_submap_dict_from_pcds(segment_pcds : list [o3d.geometry.PointCloud], add_random_bias = False):
    submap_dict = {}
    segments = []
    center_submap_xy = torch.Tensor([0., 0.])
    num_points = 0
    translation = np.array([5, 5, 0])
    rotation_matrix = R.from_rotvec((-np.pi / 18 + np.random.ranf() * 2 * np.pi / 18) * np.array([0, 0, 1])).as_matrix()
    for pcd in segment_pcds:
        if add_random_bias:
            segment = np.array(pcd.points) @ rotation_matrix + translation
        else:
            segment = np.array(pcd.points)
        segments.append(segment)
        center_submap_xy += segment.sum(axis=0)[:2]
        num_points += segment.shape[0]
    center_submap_xy /= num_points
    segment_centers = np.array([segment.mean(axis=0) - np.hstack([center_submap_xy, 0.]) for segment in segments])

    submap_dict['segment_centers'] = torch.Tensor(segment_centers)
    submap_dict['segment_scales'] = torch.Tensor(np.array([np.sqrt(segment.var(axis=0)) for segment in segments]))
    submap_dict['segments'] = [torch.Tensor((segment - segment.mean(axis=0)) / np.sqrt(segment.var(axis=0))) for segment
                               in segments]
    submap_dict['segments_original'] = segments
    return submap_dict