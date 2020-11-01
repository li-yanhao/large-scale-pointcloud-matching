
import argparse
from model.PointNetVlad.dataset import make_ptclouds_info
from model.PointNetVlad.dataset import PNVDataset, PNVDatabase
from model.PointNetVlad.PointNetVlad import PointNetVlad
from model.Birdview.loss import lazy_quadruplet_loss

import os
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
from tqdm import tqdm
import torch
import torch.nn as nn

import numpy as np
from sklearn.model_selection import train_test_split


parser = argparse.ArgumentParser(description='WayzNetVlad')
parser.add_argument('--mode', type=str, default='train', help='Mode', choices=['train', 'test'])
parser.add_argument('--batch_size', type=int, default=2, help='batch_size')
parser.add_argument('--dataset_dir', type=str, default='/media/admini/LENOVO/dataset/kitti/lidar_odometry/birdview_dataset/tmp', help='dataset_dir')
parser.add_argument('--sequence_train', type=str, default='00_ds', help='sequence_train')
parser.add_argument('--sequence_validate', type=str, default='05_ds', help='sequence_validate')

# parser.add_argument('--dataset_dir', type=str, default='/home/li/Documents/wayz/image_data/dataset', help='dataset_dir')
parser.add_argument('--num_workers', type=int, default=1, help='num_workers')
# parser.add_argument('--from_scratch', type=bool, default=True, help='from_scratch')
parser.add_argument('--pretrained_embedding', type=bool, default=False, help='pretrained_embedding')
parser.add_argument('--num_similar_neg', type=int, default=4, help='number of similar negative samples')
parser.add_argument('--margin', type=float, default=0.5, help='margin')
parser.add_argument('--use_gpu', type=bool, default=True, help='use_gpu')
parser.add_argument('--learning_rate', type=float, default=0.0005, help='learning_rate')
parser.add_argument('--positive_search_radius', type=float, default=8, help='positive_search_radius')
parser.add_argument('--negative_filter_radius', type=float, default=50, help='negative_filter_radius')
parser.add_argument('--saved_model_path', type=str,
                    default='/media/admini/lavie/dataset/birdview_dataset/saved_models', help='saved_model_path')
parser.add_argument('--epochs', type=int, default=120, help='epochs')
parser.add_argument('--load_checkpoints', type=bool, default=True, help='load_checkpoints')
parser.add_argument('--num_clusters', type=int, default=64, help='num_clusters')
parser.add_argument('--final_dim', type=int, default=256, help='final_dim')
parser.add_argument('--top_k', type=int, default=25, help='top_k')
parser.add_argument('--num_points', type=int, default=4096, help='num_points')


args = parser.parse_args()


def main():
    images_info_train = make_ptclouds_info(
        struct_filename=os.path.join(args.dataset_dir, 'struct_file_' + args.sequence_train + '.txt'))
    images_info_validate = make_ptclouds_info(
        struct_filename=os.path.join(args.dataset_dir, 'struct_file_' + args.sequence_validate + '.txt'))

    validate_images_dir = os.path.join(args.dataset_dir, args.sequence_validate)
    train_images_dir = os.path.join(args.dataset_dir, args.sequence_train)

    train_database_images_info, train_query_images_info = train_test_split(images_info_train, test_size=0.1, random_state=42)

    validate_database_images_info, validate_query_images_info = train_test_split(images_info_validate, test_size=0.2, random_state=20)
    train_dataset = PNVDataset(ptclouds_info=train_database_images_info, ptclouds_dir=train_images_dir,
                               num_similar_negatives=args.num_similar_neg,
                               positive_search_radius=args.positive_search_radius,
                               negative_filter_radius=args.negative_filter_radius,
                               add_rotation=True)
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                   num_workers=args.num_workers)

    model = PointNetVlad(global_feat=True, feature_transform=True, max_pool=False,
                       output_dim=256, num_points=args.num_points)
    model.train()


    # saved_model_file = os.path.join(args.saved_model_path, 'model-lazy-triplet.pth.tar')
    saved_model_file = os.path.join(args.saved_model_path, 'PNV-kitti00-4096.pth.tar')
    if args.load_checkpoints:
        model_checkpoint = torch.load(saved_model_file, map_location=lambda storage, loc: storage)
        model.load_state_dict(model_checkpoint)
        print("Loaded model checkpoints from \'{}\'.".format(saved_model_file))


    # optimizer = optim.Adam([base_model.parameters(), net_vlad.parameters()], lr=args.learning_rate)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # start training
    # if args.mode == 'train':
    for epoch in range(args.epochs):
        # validate(model, train_images_info, validation_images_info, writer=None)
        epoch = epoch + 1
        if epoch % 3 == 0:
            validate(model, images_info_validate, validate_query_images_info, validate_images_dir)
            # validate(model, validate_database_images_info, validate_query_images_info, validate_images_dir)
            # validate(model, train_database_images_info, train_query_images_info, train_images_dir)



        # train(epoch, model, optimizer, train_data_loader)
        # torch.save(model.state_dict(), saved_model_file)
        # print("Saved models in \'{}\'.".format(saved_model_file))


# TODO
def train(epoch, model, optimizer, train_data_loader, writer=None):
    print("Processing epoch {} ......".format(epoch))
    epoch_loss = 0
    iteration = 0
    model.train()
    device = torch.device("cuda" if args.use_gpu else "cpu")
    # device = torch.device("cpu")
    model.to(device)
    criterion = nn.TripletMarginLoss(margin=args.margin ** 0.5, p=2, reduction='sum')
    with tqdm(train_data_loader) as tq:
        for query, positive, negatives, unrelated in tq:
            iteration += 1
            optimizer.zero_grad()

            B, _, N, _ = query.shape # B, 1, N, 3
            B, nneg, _, N, _ = negatives.shape # B, nneg, 1, N, 3
            # assert npos == nneg

            # print(query.shape)  # B * C * W * H
            # print(positives.shape)  # B * npos * C * W * H
            # print(negatives.shape)  # B * nneg * C * W * H
            # print(unrelated.shape) # B * C * W * H
            input = torch.cat([query, positive, negatives.view(-1, 1, N, 3), unrelated],
                              dim=0)  # (B * (1 + npos + nneg)) * C * W * H
            # input = input.reshape(-1, C, W, H) # (B * (1 + npos + nneg)) * C * W * H
            input = input.to(device)
            vlad_encoding = model(input)  # (B * (1 + npos + nneg))(17) * dim_vlad(16384)
            # print(vlad_encoding.shape, B, npos, nneg)
            vladQ, vladP, vladN, vladU = torch.split(vlad_encoding, [B, B, B * nneg, B])
            descriptor_dim = vladQ.shape[-1]
            vladQ = vladQ.view(B, 1, descriptor_dim)
            vladP = vladP.view(B, 1, descriptor_dim)
            vladN = vladN.view(B, nneg, descriptor_dim)
            vladU = vladU.view(B, 1, descriptor_dim)


            # loss = lazy_triplet_loss(vladQ, vladP, vladN, device)
            loss = lazy_quadruplet_loss(vladQ, vladP, vladN, vladU, device)

            # loss = 0
            # for i_batch in range(B):
            #     index_offet = i_batch * npos
            #     for i in range(npos):
            #         loss += criterion(vladQ[i_batch:i_batch + 1],
            #                           vladP[index_offet + i:index_offet + i + 1],
            #                           vladN[index_offet + i:index_offet + i + 1])
            # loss /= B * npos

            loss.backward()
            optimizer.step()

            # record training loss
            epoch_loss += loss.item()
            if iteration % 50 == 0:
                print("epoch_loss: {}".format(epoch_loss / iteration))
                if writer is not None:
                    writer.add_scalar('Train/Loss', epoch_loss / iteration,
                                      (epoch * len(train_data_loader) + iteration))
                # print('Cuda memory allocated:', torch.cuda.memory_allocated() / 1024 ** 2, "MB")
                # print('Cuda memory cached:', torch.cuda.memory_reserved() / 1024 ** 2, "MB")

            del input, vlad_encoding, vladQ, vladP, vladN
            del query, positive, negatives, unrelated
    torch.cuda.empty_cache()


def validate(model, database_ptclouds_info, query_ptclouds_info, ptclouds_dir):
    model.eval()
    ptcloud_database = PNVDatabase(ptclouds_info=database_ptclouds_info, ptclouds_dir=ptclouds_dir, model=model,
                                 num_points=args.num_points, generate_database=True)
    top_k_recall = np.zeros(args.top_k)
    descriptors = []
    for query_ptcloud_info in tqdm(query_ptclouds_info):
        is_true_result = np.zeros(args.top_k)
        query_results = ptcloud_database.query_ptcloud(
            ptcloud_filename=os.path.join(ptclouds_dir, query_ptcloud_info['pcd_file']), num_results=args.top_k)
        for i, query_result in enumerate(query_results):
            diff = query_ptcloud_info['position'] - query_result['position']
            if np.sqrt(diff @ diff) < args.positive_search_radius:
                # true_count += 1
                is_true_result[i] = 1
        is_true_result = np.cumsum(is_true_result) > 0
        top_k_recall = top_k_recall + is_true_result
    # print("Precision: {}".format(true_count / len(query_ptclouds_info)))
    print("top k recalls: {}".format(top_k_recall / len(query_ptclouds_info)))


if __name__ == "__main__":
    main()