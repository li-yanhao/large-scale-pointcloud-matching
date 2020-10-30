import argparse
import os

import numpy as np
from sklearn.model_selection import train_test_split
# import Superglue.dataset.SuperglueDataset as SuperglueDataset
from torch.utils.data import DataLoader

from model.Birdview.dataset import make_images_info
from model.Superglue.dataset import SuperglueDataset
from model.Superglue.dataset import pts_from_pixel_to_meter, pts_from_meter_to_pixel
import torch
import torch.nn as nn
from tqdm import tqdm
from model.Superglue.matching import Matching
import torch.optim as optim
import visdom
# import cv2


parser = argparse.ArgumentParser(description='SuperglueTrain')
parser.add_argument('--mode', type=str, default='train', help='Mode', choices=['train', 'test'])
parser.add_argument('--batch_size', type=int, default=6, help='batch_size')
parser.add_argument('--dataset_dir', type=str, default='/media/admini/lavie/dataset/birdview_dataset/', help='dataset_dir')
parser.add_argument('--sequence_train', type=str, default='00', help='sequence_train')
parser.add_argument('--sequence_validate', type=str, default='08', help='sequence_validate')
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
    images_info_train = make_images_info(
        struct_filename=os.path.join(args.dataset_dir, 'struct_file_' + args.sequence_train + '.txt'))
    images_info_validate = make_images_info(
        struct_filename=os.path.join(args.dataset_dir, 'struct_file_' + args.sequence_validate + '.txt'))

    train_images_dir = os.path.join(args.dataset_dir, args.sequence_train)
    validate_images_dir = os.path.join(args.dataset_dir, args.sequence_validate)

    # train_database_images_info, train_query_images_info = train_test_split(images_info_train, test_size=0.1,
    #                                                                        random_state=23)
    # validate_database_images_info, validate_query_images_info = train_test_split(images_info_validate, test_size=0.2,
    #                                                                              random_state=23)

    train_dataset = SuperglueDataset(images_info=images_info_train, images_dir=train_images_dir,
                                     positive_search_radius=args.positive_search_radius,
                                     meters_per_pixel=args.meters_per_pixel)
    validate_dataset = SuperglueDataset(images_info=images_info_validate, images_dir=validate_images_dir,
                                     positive_search_radius=args.positive_search_radius,
                                     meters_per_pixel=args.meters_per_pixel)
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    validate_data_loader = DataLoader(validate_dataset, batch_size=1, shuffle=True)

    saved_model_file = os.path.join(args.saved_model_path, 'superglue-lidar-rotation-invariant.pth.tar')

    config = {
        'superpoint': {
            'nms_radius': 4,
            'keypoint_threshold': 0.005,
            'max_keypoints': 400,
        },
        'Superglue': {
            'weights': 'outdoor',
            'sinkhorn_iterations': 100,
            'match_threshold': 0.2,
        }
    }
    device = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu')
    model = Matching(config).to(device)

    if args.load_checkpoints:
        model_checkpoint = torch.load(saved_model_file, map_location=lambda storage, loc: storage)
        model.load_state_dict(model_checkpoint)
        print("Loaded model checkpoints from \'{}\'.".format(saved_model_file))

    optimizer = optim.Adam(list(model.superglue.parameters())
                           + list(model.superpoint.convDa.parameters())
                           + list(model.superpoint.convDb.parameters()),
                           lr=args.learning_rate)
    viz = visdom.Visdom()
    train_loss = viz.scatter(X=np.asarray([[0, 0]]))
    train_precision = viz.scatter(X=np.asarray([[0, 0]]))
    train_recall = viz.scatter(X=np.asarray([[0, 0]]))
    train_true_pairs = viz.scatter(X=np.asarray([[0, 0]]))
    viz_train = {
        'viz': viz,
        'train_loss': train_loss,
        'train_precision': train_precision,
        'train_recall': train_recall,
        'train_true_pairs': train_true_pairs,
    }

    viz_validate = {
        'viz': viz,
        'validate_precision': train_precision,
        'validate_recall': train_recall,
        'validate_true_pairs': train_true_pairs,
    }

    for epoch in range(args.epochs):
        epoch = epoch + 1
        if epoch % 1 == 0:
            validate(epoch, model, validate_data_loader, viz_validate=viz_validate)
            # validate_sift(sift_matcher, validate_data_loader)
        train(epoch, model, optimizer, train_data_loader, viz_train=viz_train)
        torch.save(model.state_dict(), saved_model_file)
        print("Saved models in \'{}\'.".format(saved_model_file))



    pass


def train(epoch, model, optimizer, data_loader, viz_train=None):
    print("Processing epoch {} ......".format(epoch))
    # epoch_loss = 0
    accum_loss = 0
    accum_accuracy = 0
    accum_recall = 0
    accum_precision = 0
    accum_true_pairs = 0
    print_results_period = 20
    count_accumulate = 0
    iteration = (epoch-1) * len(data_loader)

    model.train()
    device = torch.device("cuda" if args.use_gpu else "cpu")
    # device = torch.device("cpu")
    model.to(device)
    # criterion = nn.TripletMarginLoss(margin=args.margin ** 0.5, p=2, reduction='sum')
    with tqdm(data_loader) as tq:
        for targets, sources, T_target_sources in tq:
            iteration += 1
            optimizer.zero_grad()
            batch_loss = None
            for target, source, T_target_source in zip(targets, sources, T_target_sources):
                assert(target.shape == source.shape)
                C, W, H = target.shape
                target = target[None, ...].to(device)
                source = source[None, ...].to(device)
                pred = model({'image0': target, 'image1': source})
                target_kpts = pred['keypoints0'][0].cpu()
                source_kpts = pred['keypoints1'][0].cpu()
                if len(target_kpts) == 0 or len(source_kpts) == 0:
                    continue
                # in superglue/numpy/tensor the coordinates are (u,v) which correspond to (y,x)
                target_kpts_in_meters = pts_from_pixel_to_meter(target_kpts, args.meters_per_pixel)
                source_kpts_in_meters = pts_from_pixel_to_meter(source_kpts, args.meters_per_pixel)
                match_mask_ground_truth = make_ground_truth_matrix(target_kpts_in_meters, source_kpts_in_meters, T_target_source,
                                                                   args.tolerance_in_meters)

                # DEBUG:
                # N, D = source_kpts.shape
                # T_target_source = T_target_source[0]
                # source_kpts_in_meters = torch.cat([source_kpts_in_meters, torch.zeros(N, 1)], dim=1)
                #
                # source_kpts_in_meters_in_target_img = source_kpts_in_meters @  \
                #                                       (T_target_source[0:3, 0:3].transpose(1, 0).float()) + T_target_source[0:3, 3]
                # source_kpts_in_meters_in_target_img = source_kpts_in_meters_in_target_img[:,:2]
                # source_kpts_in_target_img = pts_from_meter_to_pixel(source_kpts_in_meters_in_target_img,
                #                                                     args.meters_per_pixel)
                #
                # source_kpts = np.round(source_kpts.numpy()).astype(int)
                # source_kpts_in_target_img = np.round(source_kpts_in_target_img.numpy()).astype(int)
                #
                # target_image = target[0][0].cpu().numpy()
                # source_image = source[0][0].cpu().numpy()
                # target_image = np.stack([target_image] * 3, -1) * 5
                # source_image = np.stack([source_image] * 3, -1) * 5
                #
                # for (x0, y0), (x1, y1) in zip(source_kpts, source_kpts_in_target_img):
                #     cv2.circle(source_image, (x0, y0), 2, (0, 255, 0), 1, lineType=cv2.LINE_AA)
                #     cv2.circle(target_image, (x1, y1), 2, (255, 0, 0), 1, lineType=cv2.LINE_AA)
                #
                # cv2.imshow('target_image', target_image)
                # cv2.imshow('source_image', source_image)
                #
                # cv2.waitKey(0)
                # End of DEBUG


                # print(match_mask_ground_truth[:-1,:-1].sum())

                # match_mask_ground_truth
                # matches = pred['matches0'][0].cpu().numpy()
                # confidence = pred['matching_scores0'][0].cpu().detach().numpy()

                # loss = ...
                loss = -pred['scores'][0] * match_mask_ground_truth.to(device)
                loss = loss.sum()
                if batch_loss is None:
                    batch_loss = loss
                else :
                    batch_loss += loss

                # record training loss
                if match_mask_ground_truth[:-1, :-1].sum() > 0 and (pred['matches0']>0).sum() > 0 and (pred['matches1']>0).sum()>0:
                    metrics = compute_metrics(pred['matches0'], pred['matches1'], match_mask_ground_truth)
                    accum_accuracy += float(metrics['matches0_acc'])
                    accum_recall += float(metrics['matches0_recall'])
                    accum_precision += float(metrics['matches0_precision'])
                    accum_true_pairs += match_mask_ground_truth[:-1, :-1].sum()
                    count_accumulate += 1
                accum_loss += loss.item()

            batch_loss.backward()
            optimizer.step()

            accum_loss += batch_loss.item()
            # accum_accuracy /= args.batch_size
            # accum_recall /= args.batch_size
            # accum_precision /= args.batch_size
            # accum_true_pairs /= args.batch_size



            if iteration % print_results_period == 0:
                print("loss: {}".format(accum_loss / print_results_period / args.batch_size))
                print("accuracy: {}".format(accum_accuracy / count_accumulate))
                print("precision: {}".format(accum_precision / count_accumulate))
                print("recall: {}".format(accum_recall / count_accumulate))
                print("true pairs: {}".format(accum_true_pairs / count_accumulate))

                if viz_train is not None:
                    viz_train['viz'].scatter(X=np.array([[iteration, float(accum_loss / print_results_period / args.batch_size)]]),
                                name="train-loss",
                                win=viz_train['train_loss'],
                                update="append")
                    viz_train['viz'].scatter(X=np.array([[iteration, accum_precision / count_accumulate]]),
                                name="train-precision",
                                win=viz_train['train_precision'],
                                update="append")
                    viz_train['viz'].scatter(X=np.array([[iteration, accum_recall / count_accumulate]]),
                                name="train-recall",
                                win=viz_train['train_recall'],
                                update="append")
                    viz_train['viz'].scatter(X=np.array([[iteration, accum_true_pairs / count_accumulate]]),
                                             name="train-true-pairs",
                                             win=viz_train['train_true_pairs'],
                                             update="append")
                # print('Cuda memory allocated:', torch.cuda.memory_allocated() / 1024 ** 2, "MB")
                # print('Cuda memory cached:', torch.cuda.memory_reserved() / 1024 ** 2, "MB")
                accum_loss = 0
                accum_accuracy = 0
                accum_recall = 0
                accum_precision = 0
                accum_true_pairs = 0
                count_accumulate = 0

            del target, source
    torch.cuda.empty_cache()


def validate(epoch, model, data_loader, viz_validate=None):
    torch.set_grad_enabled(False)
    iteration = (epoch - 1) * len(data_loader)
    accum_accuracy = 0
    accum_recall = 0
    accum_precision = 0
    accum_true_pairs = 0
    count_accumulate = 0

    overall_detection = 0
    overall_recall = 0
    overall_precision = 0
    overall_true_pairs = 0
    overall_count = 0


    device = torch.device("cuda" if args.use_gpu else "cpu")
    with tqdm(data_loader) as tq:
        for target, source, T_target_source in tq:
            iteration += 1
            assert(target.shape == source.shape)
            B, C, W, H = target.shape
            target = target.to(device)
            source = source.to(device)
            pred = model({'image0': target, 'image1': source})

            # comment for computation cose evaluation
            target_kpts = pred['keypoints0'][0].cpu()
            source_kpts = pred['keypoints1'][0].cpu()
            if len(target_kpts) == 0 or len(source_kpts) == 0:
                continue

            # in superglue/numpy/tensor the coordinates are (i,j) which correspond to (v,u) in PIL Image/opencv
            target_kpts_in_meters = pts_from_pixel_to_meter(target_kpts, args.meters_per_pixel)
            source_kpts_in_meters = pts_from_pixel_to_meter(source_kpts, args.meters_per_pixel)
            match_mask_ground_truth = make_ground_truth_matrix(target_kpts_in_meters, source_kpts_in_meters, T_target_source[0],
                                                               args.tolerance_in_meters)
            # print(match_mask_ground_truth[:-1,:-1].sum())

            # match_mask_ground_truth
            # matches = pred['matches0'][0].cpu().numpy()
            # confidence = pred['matching_scores0'][0].cpu().detach().numpy()
            if match_mask_ground_truth[:-1, :-1].sum() > 0 and (pred['matches0']>0).sum() > 0 and (pred['matches1']>0).sum()>0:
                metrics = compute_metrics(pred['matches0'], pred['matches1'], match_mask_ground_truth)

                accum_accuracy += float(metrics['matches0_acc'])
                accum_recall += float(metrics['matches0_recall'])
                accum_precision += float(metrics['matches0_precision'])
                accum_true_pairs += match_mask_ground_truth[:-1, :-1].sum()
                count_accumulate += 1

                overall_recall += float(metrics['matches0_recall'])
                overall_precision += float(metrics['matches0_precision'])
                overall_true_pairs += match_mask_ground_truth[:-1, :-1].sum()
                overall_detection += (len(target_kpts) + len(source_kpts)) / 2
                overall_count += 1

            if iteration % 50 == 0:
                print("accuracy: {}".format(accum_accuracy / 50))
                print("precision: {}".format(accum_precision / 50))
                print("recall: {}".format(accum_recall / 50))
                print("true pairs: {}".format(accum_true_pairs / 50))

                if viz_validate is not None:
                    viz_validate['viz'].scatter(X=np.array([[iteration, accum_precision / count_accumulate]]),
                                name="validate-precision",
                                win=viz_validate['validate_precision'],
                                update="append")
                    viz_validate['viz'].scatter(X=np.array([[iteration, accum_recall / count_accumulate]]),
                                name="validate-recall",
                                win=viz_validate['validate_recall'],
                                update="append")
                    viz_validate['viz'].scatter(X=np.array([[iteration, accum_true_pairs / count_accumulate]]),
                                             name="validate-true-pairs",
                                             win=viz_validate['validate_true_pairs'],
                                             update="append")
                # print('Cuda memory allocated:', torch.cuda.memory_allocated() / 1024 ** 2, "MB")
                # print('Cuda memory cached:', torch.cuda.memory_reserved() / 1024 ** 2, "MB")
                accum_accuracy = 0
                accum_recall = 0
                accum_precision = 0
                accum_true_pairs = 0
                count_accumulate = 0

            del target, source
    torch.cuda.empty_cache()
    torch.set_grad_enabled(True)

    print("average recall: {}".format(overall_recall / overall_count))
    print("average precision: {}".format(overall_precision / overall_count))
    print("average true pairs: {}".format(overall_true_pairs / overall_count))
    print("average detected points: {}".format(overall_detection / overall_count))



def validate_sift(sift, data_loader):
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
            target = target.squeeze()
            source = source.squeeze()

            target_kpts = sift.detect(target, None)
            source_kpts = sift.detect(source, None)

            if len(target_kpts) == 0 or len(source_kpts) == 0:
                continue
            #
            # # in superglue/numpy/tensor the coordinates are (i,j) which correspond to (v,u) in PIL Image/opencv
            # target_kpts_in_meters = pts_from_pixel_to_meter(target_kpts, args.meters_per_pixel)
            # source_kpts_in_meters = pts_from_pixel_to_meter(source_kpts, args.meters_per_pixel)
            # match_mask_ground_truth = make_ground_truth_matrix(target_kpts_in_meters, source_kpts_in_meters,
            #                                                    T_target_source[0],
            #                                                    args.tolerance_in_meters)
            # # print(match_mask_ground_truth[:-1,:-1].sum())
            #
            # # match_mask_ground_truth
            # # matches = pred['matches0'][0].cpu().numpy()
            # # confidence = pred['matching_scores0'][0].cpu().detach().numpy()
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
    pass


if __name__ == '__main__':
    # make_ground_truth_matrix_test()
    main()