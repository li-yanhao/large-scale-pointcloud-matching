import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import KNNGraph, EdgeConv
import numpy as np
import h5py
import open3d as o3d
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader

class MyDataset(Dataset):
    def __init__(self, h5py_file, is_train=True):
        ## record_path:记录图片路径及对应label的文件
        self.scans = []
        # self.normals = []
        self.poses = []
        self.is_train = is_train
        for group_id in h5py_file:
            raw_points = np.array(h5py_file[str(group_id) + "/lidar_scan"])
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(raw_points)
            pcd.voxel_down_sample(3)
            pcd.estimate_normals()
            scan = np.asarray(pcd.normals)
            normals = np.asarray(pcd.points)
            self.scans.append(np.hstack([scan, normals]))
            self.poses.append(np.array(h5py_file[str(group_id) + "/global_pose"]))
    # 获取单条数据
    def __getitem__(self, index):
        ref_pose = self.poses[index]
        src_pose = self.poses[index + 1]
        rel_pose = np.linalg.inv(ref_pose) @ src_pose
        sample = {"points_ref": self.scans[index], "points_src": self.scans[index+1], "transform_gt": rel_pose[0:3, 0:4]}
        return sample
    # 数据集长度
    def __len__(self):
        return len(self.scans) - 1

class Model(nn.Module):
    def __init__(self, k, feature_dims, emb_dims):
        super(Model, self).__init__()

        self.nng = KNNGraph(k)
        self.conv = EdgeConv(feature_dims, emb_dims, False)

    def forward(self, x):
        g = self.nng(x)
        return self.conv(g, x)


net = Model(k=20, feature_dims=3, emb_dims=128)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

f = h5py.File("/home/li/study/intelligent-vehicles/cooper-AR/large_scale_pointcloud_matching/tools/carla_scan.h5", "r")
x_0 = np.array(f["8/lidar_scan"])
x_1 = np.array(f["10/lidar_scan"])
pose_0 = np.array(f["8/global_pose"])
pose_1 = np.array(f["10/global_pose"])

# 1. Preprocessing
pcd_0 = o3d.geometry.PointCloud()
pcd_0.points = o3d.utility.Vector3dVector(x_0)
pcd_0.transform(pose_0)
pcd_0 = pcd_0.voxel_down_sample(3)

pcd_1 = o3d.geometry.PointCloud()
pcd_1.points = o3d.utility.Vector3dVector(x_1)
pcd_1.transform(pose_1)
pcd_1 = pcd_1.voxel_down_sample(3)


pcd_0.paint_uniform_color([1, 0, 0])
pcd_1.paint_uniform_color([0, 1, 0])
o3d.visualization.draw_geometries([pcd_0, pcd_1])

# 1.a Build dataset
dataset = MyDataset(f)
# 1.b Load dataset
trainLoader = DataLoader(dataset=dataset, batch_size=1,shuffle=False)

# 2. Construct model


# 3. Training



net.to(dev)
x = torch.Tensor(x)
x = x.to(dev)

y = net(x)
print(y.shape)
print(y)

