from torch.utils.data import Dataset
import h5py
import json
from sklearn.model_selection import train_test_split
import numpy as np
import torch


def create_submap_dataset(h5file: h5py.File):
    dataset = {}
    center_xy = torch.Tensor([0., 0.])
    num_points = 0
    for submap_name in h5file.keys():
        submap_dict = {}
        submap_dict['num_segments'] = np.array(h5file[submap_name + '/num_segments'])[0]
        segments = []
        for i in range(submap_dict['num_segments']):
            # submap_dict[segment_name] = np.array(h5file[submap_name + '/num_segments'])
            segment_name = submap_name + '/segment_' + str(i)
            segments.append(np.array(h5file[segment_name]))
            center_xy += segments[-1].sum(axis=0)[:2]
            num_points += segments[-1].shape[0]
        center_xy /= num_points
        segments = [torch.Tensor(segment - np.hstack([center_xy, 0.])) for segment in segments]
        submap_dict['segments'] = segments
        dataset[submap_name] = submap_dict

    return dataset


def make_match_mask_ground_truth(ids_A: np.ndarray, ids_B: np.ndarray, size_A: int, size_B: int):
    match_mask_ground_truth = np.zeros((size_A + 1, size_B + 1))
    match_mask_ground_truth[:size_A, -1:] = 1.
    match_mask_ground_truth[-1:, :size_B] = 1.
    match_mask_ground_truth[(ids_A, ids_B)] = 1.
    match_mask_ground_truth[ids_A, -1:] = 0.
    match_mask_ground_truth[-1:, ids_B] = 0.
    return torch.Tensor(match_mask_ground_truth)


class GlueNetDataset(Dataset):
    def __init__(self, submap_filename: str, correspondences_filename: str, mode):
        super(GlueNetDataset, self).__init__()

        self.mode = mode

        h5_file = h5py.File(submap_filename, 'r')
        self.num_submaps = len(h5_file.keys())

        self.dataset = create_submap_dataset(h5_file)

        with open(correspondences_filename) as f:
            correspondences_all = json.load(f)['correspondences']
            correspondences_all = [{
                'submap_pair': correspondence['submap_pair'].split(','),
                'segment_pairs': np.array(list(map(int, correspondence['segment_pairs'].split(',')[:-1]))).reshape(-1,
                                                                                                                   2).transpose(),
            } for correspondence in correspondences_all]

        correspondences_train, correspondences_test = train_test_split(correspondences_all, test_size=0.3,
                                                                       random_state=1)

        if mode == 'train':
            self.correspondences = correspondences_train
        if mode == 'test':
            self.correspondences = correspondences_test

    def __len__(self):
        return len(self.correspondences)

    def __getitem__(self, i):

        correspondence = self.correspondences[i]
        submap_ids = correspondence['submap_pair']
        submap_A_name = 'submap_' + submap_ids[0]
        submap_B_name = 'submap_' + submap_ids[1]

        # segment_id_pairs = list(map(int, correspondence['segment_pairs'].split(',')[:-1])) # remove the last empty string
        # segment_id_pairs = np.array(segment_id_pairs).reshape(-1, 2).transpose()

        segment_id_pairs = correspondence['segment_pairs']
        segments_A = self.dataset[submap_A_name]['segments']
        segments_B = self.dataset[submap_B_name]['segments']

        match_mask_ground_truth = make_match_mask_ground_truth(segment_id_pairs[0], segment_id_pairs[1], len(segments_A),
                                                            len(segments_B))

        return self.dataset[submap_A_name]['segments'], self.dataset[submap_B_name]['segments'], match_mask_ground_truth

if __name__ == "__main__":
    h5_filename = "/media/admini/My_data/0629/submap_segments.h5"
    correspondences_filename = "/media/admini/My_data/0629/correspondences.json"

    gluenet_dataset = GlueNetDataset(h5_filename, correspondences_filename, mode='train')

    print("Starting to retrieve data ... ")
    for i in range(10000):
        segments_A, segments_B, matches_ground_truth = gluenet_dataset[100]
        print("Retrieved {}-th item.".format(i))
    print("Finished. ")
