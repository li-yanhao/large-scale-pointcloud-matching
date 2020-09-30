import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import torchvision.transforms as transforms
from sklearn.neighbors import NearestNeighbors
import faiss
from tqdm import tqdm
from scipy.spatial.transform import Rotation
import h5py
import json
from sklearn.model_selection import train_test_split


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


class NetVladDataset(Dataset):
    def __init__(self, submap_filename: str, correspondences_filename: str, mode='train'):
        super(NetVladDataset, self).__init__()

        self.mode = mode

        h5_file = h5py.File(submap_filename, 'r')
        self.num_submaps = len(h5_file.keys())

        self.dataset = create_submap_dataset(h5_file)
        f = open(correspondences_filename, 'r')
        correspondences_all = json.load(f)['correspondences']
        correspondences_all = np.array([
            np.array(list(map(int, correspondence['submap_pair'].split(','))))
            for correspondence in correspondences_all
        ])



        # correspondences_all = [{
        #     'submap_pair': correspondence['submap_pair'].split(','),
        #     'segment_pairs': np.array(list(map(int, correspondence['segment_pairs'].split(',')[:-1]))).reshape(-1,
        #                                                                                                        2).transpose(),
        # } for correspondence in correspondences_all]

        # correspondences_all = [{
        #     'submap_pair': correspondence['submap_pair'].split(',')
        # } for correspondence in correspondences_all]

        correspondences_train, correspondences_test = train_test_split(correspondences_all, test_size=0.5,
                                                                       random_state=1, shuffle=True)
        f.close()
        if mode == 'train':
            self.correspondences = correspondences_train
        if mode == 'test':
            self.correspondences = correspondences_test

        self.submap_ids = np.unique(self.correspondences[:,0])

    def __len__(self):
        return len(self.submap_ids)

    def __getitem__(self, index):
        query_submap_id = self.submap_ids[index]
        positive_ids = self.correspondences[self.correspondences[:,0]==query_submap_id,1]
        negative_ids = np.random.choice(np.setdiff1d(self.submap_ids, positive_ids), len(positive_ids), replace=False)

        query = self.dataset['submap_'+str(query_submap_id)]
        positives = [self.dataset['submap_'+str(id)] for id in positive_ids]
        negatives = [self.dataset['submap_'+str(id)] for id in negative_ids]

        return query, positives, negatives
        # positives = []
        # for pos_index in self.list_of_positives_indices[index]:
        #     positives.append(Image.open(os.path.join(self.images_dir, self.images_info[pos_index]['image_file'])))
        # negatives = []
        # for neg_index in self.list_of_negative_indices[index]:
        #     negatives.append(Image.open(os.path.join(self.images_dir, self.images_info[neg_index]['image_file'])))
        #
        # if self.input_transforms:
        #     query = self.input_transforms(query)
        #     negatives = torch.cat([self.input_transforms(img).unsqueeze(0) for img in negatives])
        #     positives = torch.cat([self.input_transforms(img).unsqueeze(0) for img in positives])
        # return query, positives, negatives


# class ValidationDatabase(object):
#     def






if __name__ == "__main__":
    h5_filename = "/media/admini/My_data/0629/submap_segments.h5"
    correspondences_filename = "/media/admini/My_data/0629/correspondences.json"

    netvlad_dataset = NetVladDataset(h5_filename, correspondences_filename, mode='train')
    train_loader = DataLoader(netvlad_dataset, batch_size=1, shuffle=True)
    for item in train_loader:
        query, pos, neg = item
        print(query, pos, neg)
