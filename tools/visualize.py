#!/usr/bin/env python3
import open3d as o3d
import numpy as np

print("Testing IO for point cloud ...")
# pcd = o3d.io.read_point_cloud("../../TestData/pointcloud_1386.txt.npy.pcd")
# np_array = np.load("/media/li/LENOVO/dataset/carla_data/scans_0704/scan_2.npy")
np_array = np.load("/tmp/submap_0.npy")
# Pass xyz to Open3D.o3d.geometry.PointCloud and visualize
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(np_array)
o3d.visualization.draw_geometries([pcd])

# o3d.io.write_point_cloud("../../TestData/sync.ply", pcd)

#######################
## Open3d with Numpy ##
#######################

# Load saved point cloud and visualize it
# pcd_load = o3d.io.read_point_cloud("../../TestData/sync.ply")

# convert Open3D.o3d.geometry.PointCloud to numpy array
# xyz_load = np.asarray(pcd_load.points)
# print('xyz_load')
# print(xyz_load)
# o3d.visualization.draw_geometries([pcd_load])

#######################
##   Visualization   ##
#######################

# print("Load a ply point cloud, print it, and render it")
# pcd = o3d.io.read_point_cloud("../../TestData/fragment.ply")
# o3d.visualization.draw_geometries([pcd], zoom=0.3412,
#                                   front=[0.4257, -0.2125, -0.8795],
#                                   lookat=[2.6172, 2.0475, 1.532],
#                                   up=[-0.0694, -0.9768, 0.2024])