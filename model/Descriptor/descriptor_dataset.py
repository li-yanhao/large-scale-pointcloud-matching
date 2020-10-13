import os

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from sklearn.neighbors import NearestNeighbors
import faiss
from scipy.spatial.transform import Rotation as R
import h5py
import json
from sklearn.model_selection import train_test_split
import open3d as o3d
import matplotlib.pyplot as plt


def locate_query_in_array(query_id, arr):
    arr_cumsum = arr.cumsum()
    arr_tmp = arr_cumsum - query_id - 1
    arr_tmp[arr_tmp < 0] = 65535
    global_id = arr_tmp.argmin()
    local_id = query_id - arr_cumsum[global_id-1] if global_id > 0 else query_id

    return global_id, local_id


def create_submap_dataset(h5file: h5py.File):
    dataset = {}
    for submap_name in h5file.keys():
        submap_dict = {}
        submap_dict['num_segments'] = np.array(h5file[submap_name + '/num_segments'])[0]
        segments = []
        center_submap_xy = torch.Tensor([0., 0.])
        num_points = 0
        for i in range(submap_dict['num_segments']):
            segment_name = submap_name + '/segment_' + str(i)
            segments.append(np.array(h5file[segment_name]))
            center_submap_xy += segments[-1].sum(axis=0)[:2]
            num_points += segments[-1].shape[0]
        center_submap_xy /= num_points
        # segments = [np.array(segment - np.hstack([center_submap_xy, 0.])) for segment in segments]
        segment_centers = np.array([segment.mean(axis=0) - np.hstack([center_submap_xy, 0.]) for segment in segments])

        submap_dict['segment_centers'] = torch.Tensor(segment_centers)
        submap_dict['segment_scales'] = torch.Tensor(np.array([np.sqrt(segment.var(axis=0)) for segment in segments]))
        submap_dict['segments'] = [torch.Tensor((segment - segment.mean(axis=0)) / np.sqrt(segment.var(axis=0))) for segment in segments]

        dataset[submap_name] = submap_dict

    return dataset


def rotate_by_matrix(vectors : torch.Tensor, rotation_matrix : torch.Tensor):
    return vectors @ rotation_matrix.transpose(0, 1)


class DescriptorDataset(Dataset):
    def __init__(self, submap_filename: str, correspondences_filename: str, mode='train', random_rotate=True):
        super(DescriptorDataset, self).__init__()

        self.mode = mode
        self.random_rotate = random_rotate

        h5_file = h5py.File(submap_filename, 'r')
        self.num_submaps = len(h5_file.keys())

        self.dataset = create_submap_dataset(h5_file)
        f = open(correspondences_filename, 'r')
        correspondences_all = json.load(f)['correspondences']

        correspondences_all = [{
            'submap_pair': correspondence['submap_pair'].split(','),
            'segment_pairs': np.array(list(map(int, correspondence['segment_pairs'].split(',')[:-1]))).reshape(-1,
                                                                                                               2).transpose(),
        } for correspondence in correspondences_all]

        correspondences_train, correspondences_test = train_test_split(correspondences_all, test_size=0.5,
                                                                       random_state=1, shuffle=True)
        f.close()
        if mode == 'train':
            self.correspondences = correspondences_train
        elif mode == 'test':
            self.correspondences = correspondences_test
        else:
            print("Unknown mode: {}".format(mode))
            self.correspondences = None

        self.arr_num_segment_pairs = np.array([correspondence['segment_pairs'].shape[1] for correspondence in
                                        self.correspondences])
        self.cumsum_num_segment_paris = self.arr_num_segment_pairs.cumsum()
        self.dataset_size = self.arr_num_segment_pairs.sum()


    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index):
        correspondence_id, segment_pair_id = locate_query_in_array(index, self.arr_num_segment_pairs)
        positive_submap_ids = self.correspondences[correspondence_id]['submap_pair']
        positive_segment_ids = self.correspondences[correspondence_id]['segment_pairs'][:, segment_pair_id]

        negative_segment_ids = np.setdiff1d(np.arange(self.dataset['submap_' + positive_submap_ids[0]]['num_segments']),
                                           np.array([positive_segment_ids[0]]))
        negative_segment_id = np.random.choice(negative_segment_ids, 1)[0]

        anchor = {
            'segment': self.dataset['submap_' + positive_submap_ids[0]]['segments'][positive_segment_ids[0]],
            'segment_scale': self.dataset['submap_' + positive_submap_ids[0]]['segment_scales'][
                positive_segment_ids[0]],
            'segment_center': self.dataset['submap_' + positive_submap_ids[0]]['segment_centers'][
                positive_segment_ids[0]],
        }

        positive = {
            'segment': self.dataset['submap_' + positive_submap_ids[1]]['segments'][positive_segment_ids[1]],
            'segment_scale': self.dataset['submap_' + positive_submap_ids[1]]['segment_scales'][
                positive_segment_ids[1]],
            'segment_center': self.dataset['submap_' + positive_submap_ids[1]]['segment_centers'][
                positive_segment_ids[1]],
        }

        negative = {
            'segment': self.dataset['submap_' + positive_submap_ids[0]]['segments'][negative_segment_id],
            'segment_scale': self.dataset['submap_' + positive_submap_ids[0]]['segment_scales'][
                negative_segment_id],
            'segment_center': self.dataset['submap_' + positive_submap_ids[0]]['segment_centers'][
                negative_segment_id],
        }

        if self.random_rotate:
            rotation_matrix = torch.Tensor(
                R.from_rotvec((-np.pi + np.random.ranf() * 2 * np.pi) * np.array([0, 0, 1])).as_matrix())
            anchor['segment'] = rotate_by_matrix(anchor['segment'], rotation_matrix)
            anchor['segment_scale'] = rotate_by_matrix(anchor['segment_scale'], rotation_matrix)
            anchor['segment_center'] = rotate_by_matrix(anchor['segment_center'], rotation_matrix)

            rotation_matrix = torch.Tensor(
                R.from_rotvec((-np.pi + np.random.ranf() * 2 * np.pi) * np.array([0, 0, 1])).as_matrix())
            positive['segment'] = rotate_by_matrix(positive['segment'], rotation_matrix)
            positive['segment_scale'] = rotate_by_matrix(positive['segment_scale'], rotation_matrix)
            positive['segment_center'] = rotate_by_matrix(positive['segment_center'], rotation_matrix)

            rotation_matrix = torch.Tensor(
                R.from_rotvec((-np.pi + np.random.ranf() * 2 * np.pi) * np.array([0, 0, 1])).as_matrix())
            negative['segment'] = rotate_by_matrix(negative['segment'], rotation_matrix)
            negative['segment_scale'] = rotate_by_matrix(negative['segment_scale'], rotation_matrix)
            negative['segment_center'] = rotate_by_matrix(negative['segment_center'], rotation_matrix)

        return anchor, positive, negative


if __name__ == "__main__":
    if True:
        h5_filename = "/media/admini/My_data/submap_database/juxin-0629/submap_segments.h5"
        correspondences_filename = "/media/admini/My_data/submap_database/juxin-0629/correspondences.txt"

        descriptor_dataset = DescriptorDataset(h5_filename, correspondences_filename, mode='train', random_rotate=False)
        train_loader = DataLoader(descriptor_dataset, batch_size=1, shuffle=True)
        for item in train_loader:
            anchor, positive, negative = item

            cloud = o3d.geometry.PointCloud()
            color_labels = []
            cloud.points.extend(o3d.utility.Vector3dVector(anchor['segment'].squeeze().numpy()
                                                           * anchor['segment_scale'].squeeze().numpy()))
            color_labels.append(np.ones(anchor['segment'].shape[1]) * 1)
            cloud.points.extend(o3d.utility.Vector3dVector(positive['segment'].squeeze().numpy()
                                                           * positive['segment_scale'].squeeze().numpy()))
            color_labels.append(np.ones(positive['segment'].shape[1]) * 2)
            cloud.points.extend(o3d.utility.Vector3dVector(negative['segment'].squeeze().numpy()
                                                           * negative['segment_scale'].squeeze().numpy()))
            color_labels.append(np.ones(negative['segment'].shape[1]) * 3)
            color_labels = np.concatenate(color_labels)
            colors = plt.get_cmap("tab20")(color_labels / 4)
            cloud.colors = o3d.utility.Vector3dVector(colors[:, :3])

            o3d.visualization.draw_geometries([cloud])


