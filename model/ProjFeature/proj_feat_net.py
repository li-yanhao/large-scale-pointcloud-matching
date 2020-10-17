from model.Descriptor.descnet import *
import open3d as o3d
import numpy as np
import torch.nn as nn
import torch

if __name__ == '__main__':
    width = 100
    height = 100
    voxel_size = 0.1
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    descnet = DgcnnModel(k=4, feature_dims=[8, 32], emb_dims=[32, 8], output_classes=8)
    n_points = torch.Tensor(np.random.rand(13000, 16, 3))
    descnet.to(dev)
    n_points = n_points.to(dev)
    for i in tqdm(range(100)):
        descnet(n_points)


    # step 1: voxelize the cloud
    cloud_raw = o3d.geometry.PointCloud()
    N = 20000
    points = np.random.randn(N, 3) * np.array([width, height, 1])
    cloud_raw.points = o3d.utility.Vector3dVector(points)

    # Assume cloud_raw has been centered
    # Image width, height

    bins_width = np.arange(width / voxel_size) * voxel_size - 0.5 * width
    bins_height = np.arange(height / voxel_size) * voxel_size - 0.5 * height

    indices_width = np.digitize(points[:,0], bins_width)
    indices_height = np.digitize(points[:,1], bins_height)

    i, j = 20, 30
    indices_width == 20
    np.logical_and(indices_width == i, indices_height == j)
    #


    # o3d.visualization.draw_geometries([segments_cloud])
