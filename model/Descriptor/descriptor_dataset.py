import os

import numpy as np
import torch
<<<<<<< HEAD
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
=======
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
>>>>>>> 8ca4247be825a9b02abbb208901a86177e153943
import torchvision.transforms as transforms
from sklearn.neighbors import NearestNeighbors
import faiss
from tqdm import tqdm
from scipy.spatial.transform import Rotation
import h5py
import json
from sklearn.model_selection import train_test_split


<<<<<<< HEAD
def locate_query_in_array(query_id, arr):
    arr_cumsum = arr.cumsum()
    arr_tmp = arr_cumsum - query_id - 1
    arr_tmp[arr_tmp < 0] = 65535
    global_id = arr_tmp.argmin()
    local_id = query_id - arr_cumsum[global_id-1] if global_id > 0 else query_id

    return global_id, local_id
=======
def find_cumsum_in_array(query, arr):
    arr_cumsum = arr.cumsum()
    arr_tmp = arr_cumsum - query
    arr_tmp[arr_tmp < 0] = query
    idx = arr_tmp.argmin()
    remaining = query - arr_cumsum[idx-1] if idx > 0 else query

    return idx, remaining-1
>>>>>>> 8ca4247be825a9b02abbb208901a86177e153943


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


class DescriptorDataset(Dataset):
    def __init__(self, submap_filename: str, correspondences_filename: str, mode='train'):
        super(DescriptorDataset, self).__init__()

        self.mode = mode

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

        self.arr_num_segment_pairs = np.array([correspondence['segment_pairs'].shape[1] for correspondence in
                                  self.correspondences])
        self.cumsum_num_segment_paris = self.arr_num_segment_pairs.cumsum()
        self.dataset_size = self.arr_num_segment_pairs.sum()

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index):
<<<<<<< HEAD
        correspondence_id, segment_pair_id = locate_query_in_array(index, self.arr_num_segment_pairs)
=======
        correspondence_id, segment_pair_id = find_cumsum_in_array(index, self.arr_num_segment_pairs)
>>>>>>> 8ca4247be825a9b02abbb208901a86177e153943
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

        return anchor, positive, negative

        # query_submap_id = self.submap_ids[index]
        # positive_ids = self.correspondences[self.correspondences[:,0]==query_submap_id,1]
        # negative_ids = np.random.choice(np.setdiff1d(self.submap_ids, positive_ids), len(positive_ids), replace=False)
        #
        # query = self.dataset['submap_'+str(query_submap_id)]
        # positives = [self.dataset['submap_'+str(id)] for id in positive_ids]
        # negatives = [self.dataset['submap_'+str(id)] for id in negative_ids]
        #
        # return query, positives, negatives


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

<<<<<<< HEAD
=======

# class ValidationDatabase(object):
#     def






>>>>>>> 8ca4247be825a9b02abbb208901a86177e153943
if __name__ == "__main__":
    if True:
        h5_filename = "/media/admini/My_data/submap_database/00/submap_segments.h5"
        correspondences_filename = "/media/admini/My_data/submap_database/00/correspondences.txt"

        descriptor_dataset = DescriptorDataset(h5_filename, correspondences_filename, mode='train')
        train_loader = DataLoader(descriptor_dataset, batch_size=1, shuffle=True)
        for item in train_loader:
            anchor, positive, negative = item
