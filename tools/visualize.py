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
pcd.paint_uniform_color([1, 0.706, 0])
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


import h5py
import numpy as np
f = h5py.File("my_file.h5", "r")
x = np.array(f["MyGroup1/map/dset"])
print(x)

# f = h5py.File("/media/admini/My_data/zhonghaun_06122/partial/partial_zhonghuan.bag_segmented_submaps.h5", "r")
f = h5py.File("/media/admini/My_data/0721/zhonghuan/Paul_Zhonghuan.bag_segmented_submaps.h5", "r")
f.visit(print)


submap_id = 2
pcds = []
for submap_id in range(0, 1):
    num_segments = np.array(f["submap_" + str(submap_id) + "/num_segments"])[0]
    for i in range(num_segments):
        segment = np.array(f["submap_" + str(submap_id) + "/segment_" + str(i)])
        if segment.shape[0] > 5000:
            continue
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(segment)
        pcd.paint_uniform_color([(0.17*i+submap_id*0.7) % 1, (0.31*i+submap_id*0.7) % 1, (0.53*i+submap_id*0.7)%1])
        # pcd.paint_uniform_color(
        #     [(submap_id * 0.7 + 0.2) % 1, (submap_id * 0.5 + 0.3) % 1, (submap_id * 0.3+0.5) % 1])
        pcds.append(pcd)
o3d.visualization.draw_geometries(pcds)