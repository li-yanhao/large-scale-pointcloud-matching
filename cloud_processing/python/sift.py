import sys
sys.path.append("../../")

import cv2
import numpy as np

def demo_sift():
    img = cv2.imread('/media/admini/lavie/dataset/birdview_dataset/00/submap_100.png')
    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    print(gray.shape)
    #
    # sift = cv2.SIFT()
    # kp = sift.detect(gray, None)
    #
    #
    sift = cv2.SIFT_create(nfeatures=150, contrastThreshold=0.002, edgeThreshold=15, sigma=1.2)
    kp = sift.detect(gray,None)
    print(len(kp))
    img=cv2.drawKeypoints(gray,kp,img)

    cv2.imshow("img", img)
    cv2.waitKey(0)
    #
    #
    # # img=cv2.drawKeypoints(gray,kp)
    #
    # cv2.imwrite('sift_keypoints.png',img)


import argparse
import os

import numpy as np
from torch.utils.data import DataLoader
from model.Superglue.dataset import pts_from_meter_to_pixel, pts_from_pixel_to_meter
from model.Birdview.dataset import make_images_info
from model.Superglue.dataset import SuperglueDataset
import torch
from tqdm import tqdm
import cv2


parser = argparse.ArgumentParser(description='SuperglueTrain')
parser.add_argument('--mode', type=str, default='train', help='Mode', choices=['train', 'test'])
parser.add_argument('--batch_size', type=int, default=6, help='batch_size')
parser.add_argument('--dataset_dir', type=str, default='/media/admini/lavie/dataset/birdview_dataset/', help='dataset_dir')
parser.add_argument('--sequence_train', type=str, default='juxin_1023_map', help='sequence_train')
parser.add_argument('--sequence_validate', type=str, default='02', help='sequence_validate')
parser.add_argument('--num_workers', type=int, default=1, help='num_workers')
parser.add_argument('--use_gpu', type=bool, default=True, help='use_gpu')
parser.add_argument('--learning_rate', type=float, default=0.0003, help='learning_rate')
parser.add_argument('--positive_search_radius', type=float, default=10, help='positive_search_radius')
parser.add_argument('--saved_model_path', type=str,
                    default='/media/admini/lavie/dataset/birdview_dataset/saved_models', help='saved_model_path')
parser.add_argument('--epochs', type=int, default=100, help='epochs')
parser.add_argument('--load_checkpoints', type=bool, default=True, help='load_checkpoints')
parser.add_argument('--meters_per_pixel', type=float, default=0.25, help='meters_per_pixel')
# parser.add_argument('--tolerance_in_pixels', type=float, default=4, help='tolerance_in_pixels')
parser.add_argument('--tolerance_in_meters', type=float, default=0.5, help='tolerance_in_meters')
parser.add_argument('--number_of_features', type=int, default=130, help='number_of_features')

# parser.add_argument('--num_clusters', type=int, default=64, help='num_clusters')
# parser.add_argument('--final_dim', type=int, default=256, help='final_dim')
# parser.add_argument('--log_path', type=str, default='logs', help='log_path')
args = parser.parse_args()


# def make_ground_truth_matrix(target_kpts, source_kpts, T_target_source, pixel_tolerance=2):
#     # target_kpts: M * D (D=2 for 2D, D=3 for 3D)
#     # source_kpts: N * D
#     # T_target_source: (D+1) * (D+1)
#     M, D = target_kpts.shape
#     N, _ = source_kpts.shape
#     source_kpts_in_target = source_kpts @ T_target_source[0:2,0:2].transpose() + T_target_source[0:2,2]
#     diff = target_kpts.view(M, D, 1) - source_kpts_in_target.transpose(1,0).view(1, D, N)
#     # drop difference at z axix
#     # diff = diff[:, :(D-1), :]
#     sqr_distances = torch.einsum('mdn,mdn->mn', diff, diff)
#     match_matrix = sqr_distances < (pixel_tolerance**2)
#     return match_matrix.float()


def make_ground_truth_matrix(target_kpts, source_kpts, T_target_source, tolerance):
    # target_kpts: M * D (D=2 for 2D, D=3 for 3D)
    # source_kpts: N * D
    # T_target_source: (D+1) * (D+1)
    M, D = target_kpts.shape
    N, D = source_kpts.shape
    if D == 2:
        target_kpts_3d = torch.cat([target_kpts, torch.zeros(M, 1)], dim=1)
        source_kpts_3d = torch.cat([source_kpts, torch.zeros(N, 1)], dim=1)
    else:
        target_kpts_3d = target_kpts
        source_kpts_3d = source_kpts
    # D = 3
    source_kpts_3d_in_target = source_kpts_3d @ T_target_source[0:3,0:3].transpose(1,0).float() + T_target_source[0:3,3]
    diff = target_kpts_3d[:,:2].view(M,2,1) - source_kpts_3d_in_target[:,:2].transpose(1,0).view(1,2,N)
    sqr_distances = torch.einsum('mdn,mdn->mn', diff, diff)
    ground_truth_mask_matrix = (sqr_distances < (tolerance*tolerance)).float()
    dustbin_target = torch.ones(M) - ground_truth_mask_matrix.sum(dim=1)
    dustbin_target[dustbin_target <= 0] = 0
    dustbin_source = torch.ones(N) - ground_truth_mask_matrix.sum(dim=0)
    dustbin_source[dustbin_source <= 0] = 0
    dustbin_source = torch.cat([dustbin_source, torch.zeros(1)],dim=0)
    ground_truth_mask_matrix = torch.cat([ground_truth_mask_matrix, dustbin_target.view(M,1)], dim=1)
    ground_truth_mask_matrix = torch.cat([ground_truth_mask_matrix, dustbin_source.view(1,N+1)], dim=0)
    return ground_truth_mask_matrix
    # drop difference at z axix
    # diff = diff[:, :(D-1), :]
    # # sqr_distances = torch.einsum('mdn,mdn->mn', diff, diff)
    # match_matrix = sqr_distances < (pixel_tolerance**2)
    # return match_matrix.float()


def make_ground_truth_matrix_test():
    #################################
    # make_true_indices test #
    #################################
    N = 4
    alpha = np.random.rand() * 3.14
    rotation = np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])
    translation = np.random.randn(2, 1) * 20
    T_target_source = np.hstack([rotation, translation])
    T_target_source = np.vstack([T_target_source, np.array([0, 0, 1])])
    T_source_target = np.linalg.inv(T_target_source)
    T_target_source_se3 = np.array([[T_target_source[0,0], T_target_source[0,1], 0, T_target_source[0,2]],
                                    [T_target_source[1,0], T_target_source[1,1], 0, T_target_source[1,2]],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]])

    print("T_target_source ground truth: \n", T_target_source)
    target_points = np.random.randn(N, 2) * 500
    source_points = (T_source_target[:2, :2] @ target_points.transpose()).transpose() + T_source_target[:2, 2]
    target_points = np.vstack([target_points, np.array([1200, 1200])])
    source_points = np.vstack([source_points, np.array([1000,1000])])
    target_points = torch.Tensor(target_points)
    source_points = torch.Tensor(source_points)

    match_matrix = make_ground_truth_matrix(target_points, source_points, torch.Tensor(T_target_source_se3))
    # true_indices = make_true_indices(target_points, source_points, torch.Tensor(T_target_source_se3))
    print(match_matrix)
    pass


def compute_metrics(matches0, matches1, match_matrix_ground_truth):
    matches0 = np.array(matches0.cpu()).reshape(-1).squeeze() # M
    matches1 = np.array(matches1.cpu()).reshape(-1).squeeze() # N
    M = len(matches0)
    N = len(matches1)
    match_matrix_ground_truth = np.array(match_matrix_ground_truth.cpu()).squeeze()  # (M+1)*(N+1)

    matches0_idx_tuple = (np.arange(len(matches0)), matches0)
    matches1_idx_tuple = (matches1, np.arange(len(matches1)))

    matches0_positive_idx_tuple = (np.arange(len(matches0))[matches0>0], matches0[matches0>0])
    matches1_positive_idx_tuple = (matches1[matches1>0], np.arange(len(matches1))[matches1>0])

    # matches0_recall_idx_tuple = (np.arange(len(matches0))[match_matrix_ground_truth[:-1, -1]==0], matches0[match_matrix_ground_truth[:-1, -1]==0])
    # matches1_recall_idx_tuple = (np.arange(len(matches1))[match_matrix_ground_truth[-1, :-1]==0], matches1[match_matrix_ground_truth[-1, :-1]==0])

    # match_0_acc = match_matrix_ground_truth[:-1, :][matches0_precision_idx_tuple].mean()
    # match_1_acc = match_matrix_ground_truth.T[:-1, :][matches1_precision_idx_tuple].mean()

    metrics = {
        "matches0_acc": match_matrix_ground_truth[:, :][matches0_idx_tuple].mean(),
        "matches1_acc": match_matrix_ground_truth[:, :][matches1_idx_tuple].mean(),
        "matches0_precision": match_matrix_ground_truth[matches0_positive_idx_tuple].mean(),
        "matches1_precision": match_matrix_ground_truth[matches1_positive_idx_tuple].mean(),
        "matches0_recall": match_matrix_ground_truth[matches0_positive_idx_tuple].sum() / (M-match_matrix_ground_truth[:-1,-1].sum()),
        "matches1_recall": match_matrix_ground_truth[matches1_positive_idx_tuple].sum() / (N-match_matrix_ground_truth[-1,:-1].sum())
    }
    return metrics


def main():
    images_info_validate = make_images_info(
        struct_filename=os.path.join(args.dataset_dir, 'struct_file_' + args.sequence_validate + '.txt'))

    validate_images_dir = os.path.join(args.dataset_dir, args.sequence_validate)


    validate_dataset = SuperglueDataset(images_info=images_info_validate, images_dir=validate_images_dir,
                                     positive_search_radius=args.positive_search_radius,
                                     meters_per_pixel=args.meters_per_pixel)
    validate_data_loader = DataLoader(validate_dataset, batch_size=1, shuffle=True)

    sift = cv2.SIFT_create(nfeatures=191, contrastThreshold=0.002, edgeThreshold=15, sigma=1.2)
    orb = cv2.ORB_create(nfeatures=args.number_of_features)
    validate_detector(orb, validate_data_loader)


    pass



def validate_detector(detector, data_loader):
    accum_accuracy = 0
    accum_recall = 0
    accum_precision = 0
    accum_true_pairs = 0
    count_accumulate = 0

    overall_recall = 0
    overall_precision = 0
    overall_true_pairs = 0
    overall_count = 0

    device = torch.device("cuda" if args.use_gpu else "cpu")
    with tqdm(data_loader) as tq:
        for target, source, T_target_source in tq:
            assert (target.shape == source.shape)
            B, C, W, H = target.shape
            assert (B == 1 and C == 1)
            target = (target.squeeze().numpy() * 255).astype("uint8")
            source = (source.squeeze().numpy() * 255).astype("uint8")

            target_kpts, target_descs = detector.detectAndCompute(target, None)
            source_kpts, source_descs = detector.detectAndCompute(source, None)

            target_kpts = torch.Tensor(np.array([kp.pt for kp in target_kpts]))
            source_kpts = torch.Tensor(np.array([kp.pt for kp in source_kpts]))
            if len(target_kpts) == 0 or len(source_kpts) == 0:
                continue

            # bf = cv2.BFMatcher()
            # matches = bf.knnMatch(source_descs, target_descs, k=2)

            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

            # Match descriptors.
            matches = bf.match(source_descs, target_descs)

            # good =
            # Apply ratio test
            good = [[m.trainIdx, m.queryIdx] for m in matches]
            # for m, n in matches:
            #     if m.distance < 0.9 * n.distance:
            #         good.append([m.trainIdx, m.queryIdx])
            good = np.array(good)

            # in superglue/numpy/tensor the coordinates are (i,j) which correspond to (v,u) in PIL Image/opencv
            target_kpts_in_meters = pts_from_pixel_to_meter(target_kpts, args.meters_per_pixel)
            source_kpts_in_meters = pts_from_pixel_to_meter(source_kpts, args.meters_per_pixel)
            match_mask_ground_truth = make_ground_truth_matrix(target_kpts_in_meters, source_kpts_in_meters,
                                                               T_target_source[0],
                                                               args.tolerance_in_meters)


            if len(good) == 0 or match_mask_ground_truth[:-1,:-1].sum() == 0:
                continue

            def compute_metrics(matches, ground_truth_mask):
                TP = 0
                for target_id, source_id in matches:
                    if ground_truth_mask[target_id, source_id] > 0:
                        TP += 1
                precision = TP / len(matches)
                recall = TP / match_mask_ground_truth[:-1,:-1].sum()
                return precision, recall

            precision, recall = compute_metrics(good, match_mask_ground_truth)
            print("precision: {}".format(precision))
            print("recall: {}".format(recall))
            print("true pairs: {}".format(match_mask_ground_truth[:-1, :-1].sum()))

            overall_recall += recall
            overall_precision += precision
            overall_true_pairs += match_mask_ground_truth[:-1, :-1].sum()
            overall_count += 1

            # matches = pred['matches0'][0].cpu().numpy()
            # confidence = pred['matching_scores0'][0].cpu().detach().numpy()
            # if match_mask_ground_truth[:-1, :-1].sum() > 0 and (pred['matches0'] > 0).sum() > 0 and (
            #         pred['matches1'] > 0).sum() > 0:
            #     metrics = compute_metrics(pred['matches0'], pred['matches1'], match_mask_ground_truth)
            #
            #     accum_accuracy += float(metrics['matches0_acc'])
            #     accum_recall += float(metrics['matches0_recall'])
            #     accum_precision += float(metrics['matches0_precision'])
            #     accum_true_pairs += match_mask_ground_truth[:-1, :-1].sum()
            #     count_accumulate += 1
            #
            #     overall_recall += float(metrics['matches0_recall'])
            #     overall_precision += float(metrics['matches0_precision'])
            #     overall_true_pairs += match_mask_ground_truth[:-1, :-1].sum()
            #     overall_count += 1
    print("average precision: {}".format(overall_precision / overall_count))
    print("average recall: {}".format(overall_recall / overall_count))
    print("average true pairs: {}".format(overall_true_pairs / overall_count))
    pass


if __name__ == '__main__':
    main()
    # demo_sift()