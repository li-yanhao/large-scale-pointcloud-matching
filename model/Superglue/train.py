import argparse
import os

import numpy as np
from sklearn.model_selection import train_test_split
# import Superglue.dataset.SuperglueDataset as SuperglueDataset
from torch.utils.data import DataLoader

from model.Birdview.dataset import make_images_info
from model.Superglue.dataset import SuperglueDataset
import torch
import torch.nn as nn
from tqdm import tqdm
from model.Superglue.matching import Matching
import torch.optim as optim
import visdom


parser = argparse.ArgumentParser(description='SuperglueTrain')
parser.add_argument('--mode', type=str, default='train', help='Mode', choices=['train', 'test'])
parser.add_argument('--batch_size', type=int, default=1, help='batch_size')
parser.add_argument('--dataset_dir', type=str, default='/media/admini/lavie/dataset/birdview_dataset/', help='dataset_dir')
parser.add_argument('--sequence_train', type=str, default='00', help='sequence_train')
parser.add_argument('--sequence_validate', type=str, default='05', help='sequence_validate')
parser.add_argument('--num_workers', type=int, default=1, help='num_workers')
parser.add_argument('--use_gpu', type=bool, default=True, help='use_gpu')
parser.add_argument('--learning_rate', type=float, default=0.0005, help='learning_rate')
parser.add_argument('--positive_search_radius', type=float, default=3, help='positive_search_radius')
parser.add_argument('--saved_model_path', type=str,
                    default='/media/admini/lavie/dataset/birdview_dataset/saved_models', help='saved_model_path')
parser.add_argument('--epochs', type=int, default=120, help='epochs')
parser.add_argument('--load_checkpoints', type=bool, default=False, help='load_checkpoints')
parser.add_argument('--meters_per_pixel', type=float, default=0.2, help='meters_per_pixel')
parser.add_argument('--pixel_tolerance', type=float, default=4, help='pixel_tolerance')
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


def make_ground_truth_matrix(target_kpts, source_kpts, T_target_source, pixel_tolerance=2):
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
    source_kpts_3d_in_target = source_kpts_3d @ T_target_source[0:3,0:3].transpose(0,1).float() + T_target_source[0:3,3]
    diff = target_kpts_3d[:,:2].view(M,2,1) - source_kpts_3d_in_target[:,:2].transpose(0,1).view(1,2,N)
    sqr_distances = torch.einsum('mdn,mdn->mn', diff, diff)
    ground_truth_mask_matrix = sqr_distances < (pixel_tolerance**2)
    dustbin_target = torch.ones(M) - ground_truth_mask_matrix.sum(dim=1)
    dustbin_target[dustbin_target<0] = 0
    dustbin_source = torch.ones(N) - ground_truth_mask_matrix.sum(dim=0)
    dustbin_source[dustbin_source<0] = 0
    dustbin_source = torch.cat([dustbin_source, torch.zeros(1)],dim=0)
    ground_truth_mask_matrix = torch.cat([ground_truth_mask_matrix, dustbin_target.view(M,1)],dim=1)
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
    target_points = torch.Tensor(target_points)
    source_points = torch.Tensor(source_points)

    match_matrix = make_ground_truth_matrix(target_points, source_points, torch.Tensor(T_target_source_se3))
    # true_indices = make_true_indices(target_points, source_points, torch.Tensor(T_target_source_se3))
    print(match_matrix)
    pass


def compute_metrics(matches0, matches1, match_matrix_ground_truth):
    matches0 = np.array(matches0.cpu()).reshape(-1).squeeze() # M
    matches1 = np.array(matches1.cpu()).reshape(-1).squeeze() # N
    match_matrix_ground_truth = np.array(match_matrix_ground_truth.cpu()).squeeze()  # M*N

    matches0_idx_tuple = (np.arange(len(matches0)), matches0)
    matches1_idx_tuple = (np.arange(len(matches1)), matches1)

    matches0_precision_idx_tuple = (np.arange(len(matches0))[matches0>0], matches0[matches0>0])
    matches1_precision_idx_tuple = (np.arange(len(matches1))[matches1>0], matches1[matches1>0])

    matches0_recall_idx_tuple = (np.arange(len(matches0))[match_matrix_ground_truth[:-1, -1]==0], matches0[match_matrix_ground_truth[:-1, -1]==0])
    matches1_recall_idx_tuple = (np.arange(len(matches1))[match_matrix_ground_truth[-1, :-1]==0], matches1[match_matrix_ground_truth[-1, :-1]==0])

    # match_0_acc = match_matrix_ground_truth[:-1, :][matches0_precision_idx_tuple].mean()
    # match_1_acc = match_matrix_ground_truth.T[:-1, :][matches1_precision_idx_tuple].mean()

    metrics = {
        "matches0_acc": match_matrix_ground_truth[:-1, :][matches0_idx_tuple].mean(),
        "matches1_acc": match_matrix_ground_truth.T[:-1, :][matches1_idx_tuple].mean(),
        "matches0_precision": match_matrix_ground_truth[:-1, :][matches0_precision_idx_tuple].mean(),
        "matches1_precision": match_matrix_ground_truth.T[:-1, :][matches1_precision_idx_tuple].mean(),
        "matches0_recall": match_matrix_ground_truth[:-1, :][matches0_recall_idx_tuple].mean(),
        "matches1_recall": match_matrix_ground_truth.T[:-1, :][matches1_recall_idx_tuple].mean()
    }
    return metrics


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
                                     positive_search_radius=args.positive_search_radius,
                                     meters_per_pixel=args.meters_per_pixel)
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    saved_model_file = os.path.join(args.saved_model_path, 'superglue-lidar-birdview.pth.tar')

    config = {
        'superpoint': {
            'nms_radius': 4,
            'keypoint_threshold': 0.005,
            'max_keypoints': 1000,
        },
        'Superglue': {
            'weights': 'outdoor',
            'sinkhorn_iterations': 100,
            'match_threshold': 0.2,
        }
    }
    device = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu')
    model = Matching(config).eval().to(device)

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

    for epoch in range(args.epochs):
        # validate(model, train_images_info, validation_images_info, writer=None)
        epoch = epoch + 1
        # if epoch % 1 == 0:
        #     validate(model, validate_database_images_info, validate_query_images_info, validate_images_dir, writer=None)
        #     validate(model, train_database_images_info, train_query_images_info, train_images_dir, writer=None)
        train(epoch, model, optimizer, train_data_loader, viz_train)
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

    iteration = 0
    model.train()
    device = torch.device("cuda" if args.use_gpu else "cpu")
    # device = torch.device("cpu")
    model.to(device)
    # criterion = nn.TripletMarginLoss(margin=args.margin ** 0.5, p=2, reduction='sum')
    with tqdm(data_loader) as tq:
        for query, positive, T_target_source in tq:
            iteration += 1
            optimizer.zero_grad()
            assert(query.shape == positive.shape)
            B, C, W, H = query.shape
            query = query.to(device)
            positive = positive.to(device)
            pred = model({'image0': query, 'image1': positive})
            target_kpts = pred['keypoints0'][0].cpu()
            source_kpts = pred['keypoints1'][0].cpu()
            if len(target_kpts) == 0 or len(source_kpts) == 0:
                continue
            target_kpts -= 50 / args.meters_per_pixel
            source_kpts -= 50 / args.meters_per_pixel
            match_mask_ground_truth = make_ground_truth_matrix(target_kpts, source_kpts, T_target_source[0],
                                                               args.pixel_tolerance)
            # print(match_mask_ground_truth[:-1,:-1].sum())
            accum_true_pairs += match_mask_ground_truth[:-1,:-1].sum()
            # match_mask_ground_truth
            # matches = pred['matches0'][0].cpu().numpy()
            # confidence = pred['matching_scores0'][0].cpu().detach().numpy()

            # loss = ...
            loss = -pred['scores'][0] * match_mask_ground_truth.to(device)
            loss = loss.sum()

            loss.backward()
            optimizer.step()

            metrics = compute_metrics(pred['matches0'], pred['matches1'], match_mask_ground_truth)

            # record training loss
            accum_loss += loss.item()
            accum_accuracy += float(metrics['matches0_acc'])
            accum_recall += float(metrics['matches0_recall'])
            accum_precision += float(metrics['matches0_precision'])

            if iteration % 50 == 0:
                print("loss: {}".format(accum_loss / 50))
                print("accuracy: {}".format(accum_accuracy / 50))
                print("precision: {}".format(accum_precision / 50))
                print("recall: {}".format(accum_recall / 50))
                print("true pairs: {}".format(accum_true_pairs / 50))

                if viz_train is not None:
                    viz_train['viz'].scatter(X=np.array([[iteration, float(accum_loss / 50)]]),
                                name="train-loss",
                                win=viz_train['train_loss'],
                                update="append")
                    viz_train['viz'].scatter(X=np.array([[iteration, accum_precision / 50]]),
                                name="train-precision",
                                win=viz_train['train_precision'],
                                update="append")
                    viz_train['viz'].scatter(X=np.array([[iteration, accum_recall / 50]]),
                                name="train-recall",
                                win=viz_train['train_recall'],
                                update="append")
                    viz_train['viz'].scatter(X=np.array([[iteration, accum_true_pairs / 50]]),
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

            del query, positive
    torch.cuda.empty_cache()


if __name__ == '__main__':
    # make_ground_truth_matrix_test()
    main()