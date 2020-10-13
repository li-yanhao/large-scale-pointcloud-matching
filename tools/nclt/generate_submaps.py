# !/usr/bin/python
#
# Example code to read and plot the ground truth data.
#
# Note: The ground truth data is provided at a high rate of about 100 Hz. To
# generate this high rate ground truth, a SLAM solution was used. Nodes in the
# SLAM graph were not added at 100 Hz, but rather about every 8 meters. In
# between the nodes in the SLAM graph, the odometry was used to interpolate and
# provide a high rate ground truth. If precise pose is desired (e.g., for
# accumulating point clouds), then we recommend using only the ground truth
# poses that correspond to the nodes in the SLAM graph. This can be found by
# inspecting the timestamps in the covariance file.
#
# To call:
#
#   python read_ground_truth.py groundtruth.csv covariance.csv
#

import sys
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate
import struct
import open3d as o3d
import argparse
import os
from scipy.spatial.transform import Rotation as R



parser = argparse.ArgumentParser(description='nclt_dataset_to_submaps')
parser.add_argument('--velodyne_sync_dir', type=str,
                    default="/media/admini/My_data/nclt/2012-01-08/velodyne_sync", help='velodyne_sync_dir')
parser.add_argument('--ground_truth_file', type=str,
                    default='/media/admini/My_data/nclt/2012-01-08/groundtruth_2012-01-08.csv', help='ground_truth_file')
parser.add_argument('--out_submaps_dir', type=str,
                    default="/media/admini/My_data/nclt/2012-01-08/submaps", help='out_submaps_dir')
parser.add_argument('--size_submap', type=int, default=30, help='size_submap')
parser.add_argument('--ground_height', type=float, default=1.5, help='ground_height')
args = parser.parse_args()


def convert(x_s, y_s, z_s):

    scaling = 0.005 # 5 mm
    offset = -100.0

    x = x_s * scaling + offset
    y = y_s * scaling + offset
    z = z_s * scaling + offset

    return x, y, z


def ssc_to_homo(ssc):

    # Convert 6-DOF ssc coordinate transformation to 4x4 homogeneous matrix
    # transformation

    sr = np.sin(np.pi/180.0 * ssc[3])
    cr = np.cos(np.pi/180.0 * ssc[3])

    sp = np.sin(np.pi/180.0 * ssc[4])
    cp = np.cos(np.pi/180.0 * ssc[4])

    sh = np.sin(np.pi/180.0 * ssc[5])
    ch = np.cos(np.pi/180.0 * ssc[5])

    H = np.zeros((4, 4))

    H[0, 0] = ch*cp
    H[0, 1] = -sh*cr + ch*sp*sr
    H[0, 2] = sh*sr + ch*sp*cr
    H[1, 0] = sh*cp
    H[1, 1] = ch*cr + sh*sp*sr
    H[1, 2] = -ch*sr + sh*sp*cr
    H[2, 0] = -sp
    H[2, 1] = cp*sr
    H[2, 2] = cp*cr

    H[0, 3] = ssc[0]
    H[1, 3] = ssc[1]
    H[2, 3] = ssc[2]

    H[3, 3] = 1

    return H

def main(args):

    gt = np.loadtxt(args.ground_truth_file, delimiter = ",")

    # Note: Interpolation is not needed, this is done as a convenience
    interp = scipy.interpolate.interp1d(gt[:, 0], gt[:, 1:], kind='nearest', axis=0)

    # NED (North, East Down)
    # x = gt[:, 1]
    # y = gt[:, 2]
    # z = gt[:, 3]
    #
    # r = gt[:, 4]
    # p = gt[:, 5]
    # h = gt[:, 6]

    entries = os.listdir(args.velodyne_sync_dir)
    entries.sort()

    num_scans_accumulated = 0
    submap_cloud = o3d.geometry.PointCloud()
    submap_id = 0
    for cloud_filename in entries:
        timestamp = float(cloud_filename[:-4])
        pose_gt = None
        try:
            # e, n, u, r, p, h
            pose_gt = interp(timestamp)
        except:
            print('nope')
        if pose_gt is None:
            continue

        # e, n, u, r, p, h = pose_gt
        # x = n - 107.724666286
        # y = e - 75.829339527800
        # z = -u + 3.272894625813646

        x, y, z, r, p, h = pose_gt
        # T_w_body = ssc_to_homo([x, y, z, r, p, h])

        r = (R.from_euler('xyz', [r, p, h], degrees=False)).as_matrix()
        p = [x, y, z]
        n = [r[0, 0], r[1, 0], r[2, 0]]
        o = [r[0, 1], r[1, 1], r[2, 1]]
        a = [r[0, 2], r[1, 2], r[2, 2]]

        T_normal_w = np.array([[0, 1, 0, 0],
                                    [1, 0, 0, 0],
                                    [0, 0,-1, 0],
                                    [0, 0, 0, 1]])
        T_w_body = np.array([[n[0], o[0], a[0], p[0]],
                             [n[1], o[1], a[1], p[1]],
                             [n[2], o[2], a[2], p[2]],
                             [0, 0, 0, 1]])
        T_normal_body = T_normal_w @ T_w_body
        # T = np.matrix([[n[0], n[1], n[2], -np.dot(p, n)],
        #                [o[0], o[1], o[2], -np.dot(p, o)],
        #                [a[0], a[1], a[2], -np.dot(p, a)],
        #                [0, 0, 0, 1]])


        # optimized: convert point cloud to world frame
        dt = np.dtype([('x', '<H'), ('y', '<H'), ('z', '<H'), ('i', 'B'), ('l', 'B')])
        data = np.fromfile(os.path.join(args.velodyne_sync_dir, cloud_filename), dtype=dt)

        scaling = 0.005  # 5 mm
        offset = -100.0
        hits = np.vstack([data['x'], data['y'], data['z']]) * scaling + offset
        hits = np.vstack([hits, np.ones(len(data))])

        # remove ground points
        hits = hits.transpose()
        # z axis in body frame points DOWN
        hits = hits[hits[:,2] < -args.ground_height]
        hits = hits.transpose()

        hits_normal = T_normal_body @ hits
        hits_normal = hits_normal.transpose()[:, :3]

        # official: convert point cloud to world frame
        # f_bin = open(os.path.join(args.velodyne_sync_dir, cloud_filename), "br")
        #
        # hits = []
        #
        # while True:
        #
        #     x_str = f_bin.read(2)
        #     if x_str == b'':  # eof
        #         break
        #
        #     x = struct.unpack('<H', x_str)[0]
        #     y = struct.unpack('<H', f_bin.read(2))[0]
        #     z = struct.unpack('<H', f_bin.read(2))[0]
        #     i = struct.unpack('B', f_bin.read(1))[0]
        #     l = struct.unpack('B', f_bin.read(1))[0]
        #
        #     x, y, z = convert(x, y, z)
        #
        #     s = "%5.3f, %5.3f, %5.3f, %d, %d" % (x, y, z, i, l)
        #
        #     hits += [[x, y, z, 1]]
        #
        # f_bin.close()
        #
        # hits = np.asarray(hits)
        # hits_normal = T_normal_body @ hits.transpose()
        # # hits_w = np.matmul(T_w_body, hits.transpose())
        # hits_normal = hits_normal.transpose()[:,:3]

        num_scans_accumulated += 1
        submap_cloud.points.extend(o3d.utility.Vector3dVector(hits_normal))
        if num_scans_accumulated >= args.size_submap:
            submap_cloud = submap_cloud.voxel_down_sample(voxel_size=0.05)
            o3d.io.write_point_cloud(os.path.join(args.out_submaps_dir, "submap_" + str(submap_id) + ".pcd"), submap_cloud)
            submap_cloud = o3d.geometry.PointCloud()
            num_scans_accumulated = 0
            print("Added submap ", os.path.join(args.out_submaps_dir, "submap_" + str(submap_id) + ".pcd"))
            submap_id += 1

    return 0


if __name__ == '__main__':
    sys.exit(main(args))
