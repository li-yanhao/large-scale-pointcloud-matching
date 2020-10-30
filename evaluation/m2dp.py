import scipy.io as scio
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve


m2dp_kitti02_file = '/home/admini/yanhao/large-scale-pointcloud-matching/m2dp-kitti02.mat'
m2dp_kitti02 = scio.loadmat(m2dp_kitti02_file)['descriptors']



def compute_AUC(descriptors, positions, thres_distance):
    """
    decriptors: N * D
    positions: N * 3
    """
    descriptors = torch.Tensor(descriptors)
    diff = descriptors[..., None] - descriptors.transpose(0, 1)[None, ...]
    score_matrix = (1 - torch.einsum('mdn,mdn->mn', diff, diff)).numpy()
    positions = torch.cat(positions).view(-1, 3)
    diff = positions[..., None] - positions.transpose(0, 1)[None, ...]
    label_matrix = (torch.einsum('mdn,mdn->mn', diff, diff) < (thres_distance ** 2)).numpy()

    auc_score = roc_auc_score(label_matrix.reshape(-1), score_matrix.reshape(-1))

    return auc_score

print(m2dp_kitti02)