import argparse
import os

import numpy as np
from sklearn.model_selection import train_test_split
# import Superglue.dataset.SuperglueDataset as SuperglueDataset
from torch.utils.data import DataLoader

from model.Birdview.dataset import make_images_info
from model.Superglue.dataset import SuperglueDataset
import torch


parser = argparse.ArgumentParser(description='SuperglueTrain')
parser.add_argument('--mode', type=str, default='train', help='Mode', choices=['train', 'test'])
parser.add_argument('--batch_size', type=int, default=2, help='batch_size')
parser.add_argument('--dataset_dir', type=str, default='/media/admini/lavie/dataset/birdview_dataset/', help='dataset_dir')
parser.add_argument('--sequence_train', type=str, default='00', help='sequence_train')
parser.add_argument('--sequence_validate', type=str, default='05', help='sequence_validate')
parser.add_argument('--num_workers', type=int, default=1, help='num_workers')
parser.add_argument('--use_gpu', type=bool, default=True, help='use_gpu')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning_rate')
parser.add_argument('--positive_search_radius', type=float, default=7.5, help='positive_search_radius')
parser.add_argument('--saved_model_path', type=str,
                    default='/media/admini/lavie/dataset/birdview_dataset/saved_models', help='saved_model_path')
parser.add_argument('--epochs', type=int, default=120, help='epochs')
parser.add_argument('--load_checkpoints', type=bool, default=True, help='load_checkpoints')
# parser.add_argument('--num_clusters', type=int, default=64, help='num_clusters')
# parser.add_argument('--final_dim', type=int, default=256, help='final_dim')
# parser.add_argument('--log_path', type=str, default='logs', help='log_path')
args = parser.parse_args()


def make_ground_truth_matrix(target_kpts, source_kpts, T_target_source, pixel_tolerance=2):
    # target_kpts: M * D (D=2 for 2D, D=3 for 3D)
    # source_kpts: N * D
    # T_target_source: (D+1) * (D+1)
    M, D = target_kpts.shape
    N, _ = source_kpts.shape
    source_kpts_in_target = source_kpts @ T_target_source[0:2,0:2].transpose() + T_target_source[0:2,2]
    diff = target_kpts.view(M, D, 1) - source_kpts_in_target.transpose(1,0).view(1, D, N)
    sqr_distances = torch.einsum('mdn,mdn->mn', diff, diff)
    match_matrix = sqr_distances < (pixel_tolerance**2)
    return match_matrix.float()


def make_ground_truth_matrix_test():
    #################################
    # make_ground_truth_matrix test #
    #################################
    N = 4
    alpha = np.random.rand() * 3.14
    rotation = np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])
    translation = np.random.randn(2, 1) * 20
    T_target_source = np.hstack([rotation, translation])
    T_target_source = np.vstack([T_target_source, np.array([0, 0, 1])])
    T_source_target = np.linalg.inv(T_target_source)

    print("T_target_source ground truth: \n", T_target_source)
    target_points = np.random.randn(N, 2) * 500
    source_points = (T_source_target[:2, :2] @ target_points.transpose()).transpose() + T_source_target[:2, 2]
    target_points = torch.Tensor(target_points)
    source_points = torch.Tensor(source_points)
    match_matrix = make_ground_truth_matrix(target_points, source_points, T_target_source)
    print(match_matrix)


def train():
    pass


def main():
    images_info_train = make_images_info(
        struct_filename=os.path.join(args.dataset_dir, 'struct_file_' + args.sequence_train + '.txt'))
    images_info_validate = make_images_info(
        struct_filename=os.path.join(args.dataset_dir, 'struct_file_' + args.sequence_validate + '.txt'))

    train_images_dir = os.path.join(args.dataset_dir, args.sequence_train)
    validate_images_dir = os.path.join(args.dataset_dir, args.sequence_validate)

    train_database_images_info, train_query_images_info = train_test_split(images_info_train, test_size=0.1,
                                                                           random_state=23)
    validate_database_images_info, validate_query_images_info = train_test_split(images_info_validate, test_size=0.2,
                                                                                 random_state=23)

    train_dataset = SuperglueDataset(images_info=images_info_train, images_dir=train_images_dir,
                                     positive_search_radius=args.positive_search_radius)
    train_data_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)


    pass


if __name__ == '__main__':
    # make_ground_truth_matrix_test()
    main()