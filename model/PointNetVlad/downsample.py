import open3d as o3d
import os

# 00: 0-1512
# 02: 0-1552
# 05:
# 08: 0-1356
in_dir = '/media/admini/LENOVO/dataset/kitti/lidar_odometry/birdview_dataset/tmp/08_ds'
out_dir = '/media/admini/LENOVO/dataset/kitti/lidar_odometry/birdview_dataset/tmp/08_ds'
for i in range(1357):
    filename = os.path.join(in_dir, 'submap_' + str(i) + '.pcd')
    pcd = o3d.io.read_point_cloud(filename)
    pcd = pcd.uniform_down_sample(3)
    out_filename = os.path.join(out_dir, 'submap_' + str(i) + '.pcd')
    o3d.io.write_point_cloud(out_filename, pcd)
    if i % 100 == 0:
        print("processed {}-th cloud".format(i))


#
# in_dir = '/media/admini/LENOVO/dataset/kitti/lidar_odometry/birdview_dataset/tmp/02'
# out_dir = '/media/admini/LENOVO/dataset/kitti/lidar_odometry/birdview_dataset/tmp/02_ds'
# for i in range(1553):
#     filename = os.path.join(in_dir, 'submap_' + str(i) + '.pcd')
#     pcd = o3d.io.read_point_cloud(filename)
#     pcd = pcd.uniform_down_sample(3)
#     out_filename = os.path.join(out_dir, 'submap_' + str(i) + '.pcd')
#     o3d.io.write_point_cloud(out_filename, pcd)
#     if i % 100 == 0:
#         print("processed {}-th cloud".format(i))