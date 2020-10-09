from torch.utils.data import Dataset
import h5py
import json
from sklearn.model_selection import train_test_split
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

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


def make_match_mask_ground_truth(ids_A: np.ndarray, ids_B: np.ndarray, size_A: int, size_B: int):
    match_mask_ground_truth = np.zeros((size_A + 1, size_B + 1))
    match_mask_ground_truth[:size_A, -1:] = 1.
    match_mask_ground_truth[-1:, :size_B] = 1.
    match_mask_ground_truth[(ids_A, ids_B)] = 1.
    match_mask_ground_truth[ids_A, -1:] = 0.
    match_mask_ground_truth[-1:, ids_B] = 0.

    return torch.Tensor(match_mask_ground_truth)


def random_rotate(points : torch.Tensor):
    # points: N*3, torch.Tensor
    rotation_matrix = R.from_rotvec((-np.pi / 1 + np.random.ranf() * 2 * np.pi / 1) * np.array([0, 0, 1])).as_matrix()
    return points @ torch.Tensor(rotation_matrix.transpose())


def rotate_by_matrix(vectors : torch.Tensor, rotation_matrix : torch.Tensor):
    return vectors @ rotation_matrix.transpose(0, 1)


class GlueNetDataset(Dataset):
    def __init__(self, submap_filename: str, correspondences_filename: str, mode):
        super(GlueNetDataset, self).__init__()

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

        correspondences_train, correspondences_test = train_test_split(correspondences_all, test_size=0.5,
                                                                       random_state=1, shuffle=True)
        f.close()
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
        print("submaps {} and {}".format(submap_ids[0], submap_ids[1]))

        # segment_id_pairs = list(map(int, correspondence['segment_pairs'].split(',')[:-1])) # remove the last empty string
        # segment_id_pairs = np.array(segment_id_pairs).reshape(-1, 2).transpose()

        segment_id_pairs = correspondence['segment_pairs']
        # create a random rotation matrix
        rotation_matrix = torch.Tensor(
            R.from_rotvec((-np.pi / 3 + np.random.ranf() * 2 * np.pi / 3) * np.array([0, 0, 1])).as_matrix())

        segments_A = [rotate_by_matrix(segment, rotation_matrix) for segment in self.dataset[submap_A_name]['segments']]
        segments_B = self.dataset[submap_B_name]['segments']

        match_mask_ground_truth = make_match_mask_ground_truth(segment_id_pairs[0], segment_id_pairs[1], len(segments_A),
                                                            len(segments_B))

        segment_centers_A = rotate_by_matrix(self.dataset[submap_A_name]['segment_centers'], rotation_matrix)
        segment_scales_A = rotate_by_matrix(self.dataset[submap_A_name]['segment_scales'], rotation_matrix)
        segment_centers_B = self.dataset[submap_B_name]['segment_centers']
        segment_scales_B = self.dataset[submap_B_name]['segment_scales']
        return torch.cat([segment_centers_A, segment_scales_A], dim=1), \
               torch.cat([segment_centers_B, segment_scales_B], dim=1), \
               segments_A, segments_B, match_mask_ground_truth

if __name__ == "__main__":
    h5_filename = "/home/li/Documents/submap_segments.h5"
    correspondences_filename = "/home/li/Documents/correspondences.json"

    gluenet_dataset = GlueNetDataset(h5_filename, correspondences_filename, mode='train')

    print("Starting to retrieve data ... ")
    for i in range(10000):
        centers_A, centers_B, segments_A, segments_B, matches_ground_truth = gluenet_dataset[100]
        print("Retrieved {}-th item.".format(i))
    print("Finished. ")
