#!/usr/bin/env python3

import numpy as np
import h5py
import os
from sensor_msgs import point_cloud2
import rosbag
import open3d as o3d
import copy

def save_to_h5(submap, file):
    f = h5py.File(file, "w")
    # f.create_dataset(name="scans", data=map_to_export)

def get_rotation_matrix_from_quaternion(quaternion):
    w, x, y, z = quaternion
    R_data = [1-2*y*y-2*z*z, 2*x*y-2*z*w,   2*x*z+2*y*w, \
              2*x*y+2*z*w,   1-2*x*x-2*z*z, 2*y*z-2*x*w, \
              2*x*z-2*y*w,   2*y*z+2*x*w,   1-2*x*x-2*y*y]
    R = np.array(R_data)
    return np.array(R_data).reshape(3,3)


def get_transform_matrix(quaternion, position):
    R = get_rotation_matrix_from_quaternion(quaternion)
    T = np.hstack([R, position.reshape(3,1)])
    T = np.vstack([T, np.array([0,0,0,1])])
    return T


def create_lidar_scan_dataset():
    bag = rosbag.Bag('/media/li/LENOVO/dataset/carla_data/carla_2020-07-07-01-42-29.bag')
    # T_car_lidar = np.array([[0,1,0,0],[1,0,0,0],[0,0,1,0],[0,0,0,1]])
    submap = None
    global_map = None
    submap_id = 0
    scan_count = 0
    for topic, msg, t in bag.read_messages():
        if topic == "/car_1/velodyne_points":
            scan_count = scan_count + 1
            gen = point_cloud2.read_points(msg)
            data = []
            for p in gen:
                data.append(p[0])
                data.append(p[1])
                data.append(p[2])
            pc = np.array(data).reshape(-1,3)
            if submap is None:
                submap = pc
            else:
                submap = np.vstack((submap, pc))
            if scan_count % 10 == 0:
                scan_count = 0
                save_to_h5(submap, "submap_%f.h5" % submap_id )
                if global_map == None:
                    global_map = submap
                else:
                    submap = np.vstack((global_map, submap))
                submap_id = submap_id + 1

    pass


def create_lidar_scan_dataset2():
    bag = rosbag.Bag('/media/li/LENOVO/dataset/carla_data/07-13/carla_2020-07-14-01-44-31.bag')
    submap = None
    global_map = None
    submap_id = 0
    scan_count = 0
    pose_buffer = []
    pc_buffer = []
    for topic, msg, t in bag.read_messages():
        if topic == "/car_1/lidar_pose":
            # print("lidar pose t=", msg.header.stamp.to_sec())
            if scan_count >= 10:
                quaternion = np.array([msg.pose.pose.orientation.w, msg.pose.pose.orientation.x, \
                                        msg.pose.pose.orientation.y, msg.pose.pose.orientation.z]) # quaternion (w,x,y,z)
                position = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z])
                T_w_lidar = get_transform_matrix(quaternion, position)
                T_lidar_w = np.linalg.inv(T_w_lidar)

                scan_count = 0
                submap = np.array(pc_buffer).reshape(-1,3)
                submap = submap @ T_lidar_w[0:3, 0:3].T
                submap = submap + T_lidar_w[0:3, 3].reshape(3)
                # submap = (submap - T_w_lidar[0:3, 3].reshape(3)) @ T_w_lidar[0:3, 0:3]
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(submap)
                # downpcd = o3d.open3d.geometry.voxel_down_sample(pcd, voxel_size=0.2)
                # print("T_w_lidar: \n", T_w_lidar, "\n")
                # print("t= %f %f %f" % (msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z))
                
                # downpcd = copy.deepcopy(downpcd).transform(T)
                # downpcd = pcd.voxel_down_sample(voxel_size=0.05)
                o3d.io.write_point_cloud("/tmp/carla_map_%d.pcd" % submap_id, pcd)
                submap_id = submap_id + 1
                pc_buffer = []

        if topic == "/car_1/velodyne_points":
            # print("pointcloud t=", msg.header.stamp.to_sec())
            scan_count = scan_count + 1
            gen = point_cloud2.read_points(msg)
            for p in gen:
                pc_buffer.append(p[0])
                pc_buffer.append(p[1])
                pc_buffer.append(p[2])


def create_lidar_scan_h5():
    bag = rosbag.Bag('/media/li/LENOVO/dataset/carla_data/07-13/carla_2020-07-14-01-44-31.bag')
    submap = None
    lidar_scans = []
    global_poses = []
    submap_id = 0
    scan_count = 0
    pose_buffer = []
    pc_buffer = []
    for topic, msg, t in bag.read_messages():
        if topic == "/car_1/lidar_pose":
            # print("lidar pose t=", msg.header.stamp.to_sec())
            if scan_count >= 10:
                quaternion = np.array([msg.pose.pose.orientation.w, msg.pose.pose.orientation.x, \
                                        msg.pose.pose.orientation.y, msg.pose.pose.orientation.z]) # quaternion (w,x,y,z)
                position = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z])
                T_w_lidar = get_transform_matrix(quaternion, position)
                T_lidar_w = np.linalg.inv(T_w_lidar)

                scan_count = 0
                submap = np.array(pc_buffer).reshape(-1,3)
                submap = submap @ T_lidar_w[0:3, 0:3].T
                submap = submap + T_lidar_w[0:3, 3].reshape(3)
                # submap = (submap - T_w_lidar[0:3, 3].reshape(3)) @ T_w_lidar[0:3, 0:3]
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(submap)
                downpcd = o3d.open3d.geometry.voxel_down_sample(pcd, voxel_size=0.2)
                # print("T_w_lidar: \n", T_w_lidar, "\n")
                # print("t= %f %f %f" % (msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z))
                
                # downpcd = copy.deepcopy(downpcd).transform(T)
                # lidar_scans.append(submap.copy())
                lidar_scans.append(np.array(np.asarray(downpcd.points)))
                
                print("lidar_scans[0]: ", lidar_scans[0].shape) 
                global_poses.append(T_w_lidar)

                # submap_id = submap_id + 1
                pc_buffer = []

        if topic == "/car_1/velodyne_points":
            # print("pointcloud t=", msg.header.stamp.to_sec())
            scan_count = scan_count + 1
            gen = point_cloud2.read_points(msg)
            for p in gen:
                pc_buffer.append(p[0])
                pc_buffer.append(p[1])
                pc_buffer.append(p[2])
    # print("lidar_scans: ", len(lidar_scans))
    # print("lidar_scans[-1]: ", lidar_scans[-1].shape)
    # # lidar_scans_export = np.array(lidar_scans)
    # print("np.array(lidar_scans): ", np.array(lidar_scans).shape)
    # # global_poses = np.array(global_poses)
    # print("lidar_scans_export:", lidar_scans.shape)
    # print("global_poses:", global_poses.shape)
    f = h5py.File("carla_scan.h5", "w")

    for i, (pc, pose) in enumerate(zip(lidar_scans, global_poses)):
        grp = f.create_group(str(i))
        grp.create_dataset("lidar_scan", data=pc)
        grp.create_dataset("global_pose", data=pose)   
    f.flush()


if __name__== "__main__":
    # quaternion = np.array([0.8, 0.6, 0, 0])
    # position = np.array([5,2,4])
    # print(get_transform_matrix(quaternion, position))
    # create_lidar_scan_dataset2()
    create_lidar_scan_h5()

# bag = rosbag.Bag('/media/li/LENOVO/dataset/carla_data/07-13/carla_2020-07-14-01-44-31.bag')
# submap = None
# lidar_scans = []
# global_poses = []
# submap_id = 0
# scan_count = 0
# pose_buffer = []
# pc_buffer = []
# for topic, msg, t in bag.read_messages():
#     if topic == "/car_1/lidar_pose":
#         # print("lidar pose t=", msg.header.stamp.to_sec())
#         if scan_count >= 10:
#             quaternion = np.array([msg.pose.pose.orientation.w, msg.pose.pose.orientation.x, \
#                                     msg.pose.pose.orientation.y, msg.pose.pose.orientation.z]) # quaternion (w,x,y,z)
#             position = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z])
#             T_w_lidar = get_transform_matrix(quaternion, position)
#             T_lidar_w = np.linalg.inv(T_w_lidar)
#             scan_count = 0
#             submap = np.array(pc_buffer).reshape(-1,3)
#             submap = submap @ T_lidar_w[0:3, 0:3].T
#             submap = submap + T_lidar_w[0:3, 3].reshape(3)
#             # submap = (submap - T_w_lidar[0:3, 3].reshape(3)) @ T_w_lidar[0:3, 0:3]
#             pcd = o3d.geometry.PointCloud()
#             pcd.points = o3d.utility.Vector3dVector(submap)
#             downpcd = o3d.open3d.geometry.voxel_down_sample(pcd, voxel_size=0.2)
#             # print("T_w_lidar: \n", T_w_lidar, "\n")
#             # print("t= %f %f %f" % (msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z))
#             # downpcd = copy.deepcopy(downpcd).transform(T)
#             lidar_scans.append(np.asarray(downpcd.points))
#             global_poses.append(T_w_lidar)
#             submap_id = submap_id + 1
#             pc_buffer = []
#             break
#     if topic == "/car_1/velodyne_points":
#         # print("pointcloud t=", msg.header.stamp.to_sec())
#         scan_count = scan_count + 1
#         gen = point_cloud2.read_points(msg)
#         for p in gen:
#             pc_buffer.append(p[0])
#             pc_buffer.append(p[1])
#             pc_buffer.append(p[2])

