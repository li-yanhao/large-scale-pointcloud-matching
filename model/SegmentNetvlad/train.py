import os
import sys
sys.path.append("../")
import argparse
from dataset import *
from torch.utils.data import DataLoader
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.autograd import Variable
from netvlad import NetVLAD, ContrastiveLoss, EmbedNet
from descnet import DgcnnModel
# from hard_triplet_loss import HardTripletLoss
from torchvision.models import resnet18, vgg16
from PIL import Image
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from datetime import datetime
import numpy as np
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(description='SegmentNetvlad')
parser.add_argument('--mode', type=str, default='train', help='Mode', choices=['train', 'test'])
parser.add_argument('--h5_filename', type=str, default="/media/admini/My_data/0629/submap_segments.h5", help='h5_filename')
parser.add_argument('--correspondences_json', type=str, default="/media/admini/My_data/0629/correspondences.json", help='correspondences_json')
# batch_size can only be 1 now
parser.add_argument('--batch_size', type=int, default=1, help='batch_size')
parser.add_argument('--load_checkpoints', type=bool, default=False, help='load_checkpoints')
parser.add_argument('--descriptor_dim', type=int, default=128, help='descriptor_dim')
parser.add_argument('--saved_model_path', type=str, default='saved_model', help='saved_model')
# parser.add_argument('--dataset_dir', type=str, default='/media/admini/My_data/0921/dataset_cam4', help='dataset_dir')
# # parser.add_argument('--dataset_dir', type=str, default='/home/li/Documents/wayz/image_data/dataset', help='dataset_dir')
# parser.add_argument('--num_workers', type=int, default=1, help='num_workers')
# # parser.add_argument('--from_scratch', type=bool, default=True, help='from_scratch')
# parser.add_argument('--pretrained_embedding', type=bool, default=True, help='pretrained_embedding')
# parser.add_argument('--num_similar_neg', type=int, default=2, help='number of similar negative samples')
parser.add_argument('--margin', type=float, default=1.0, help='margin')
parser.add_argument('--num_clusters', type=int, default=64, help='num_clusters')
# parser.add_argument('--use_gpu', type=bool, default=True, help='use_gpu')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning_rate')
# parser.add_argument('--positive_search_radius', type=float, default=1, help='positive_search_radius')
# parser.add_argument('--negative_filter_radius', type=float, default=5, help='negative_filter_radius')

# parser.add_argument('--epochs', type=int, default=10, help='epochs')
# parser.add_argument('--num_clusters', type=int, default=64, help='num_clusters')
parser.add_argument('--log_path', type=str, default='logs', help='log_path')
args = parser.parse_args()


def main():
    train_dataset = NetVladDataset(args.h5_filename, args.correspondences_json, mode='train')

    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # database_dataset = ValidationDataset(images_info=train_images_info)
    # database_data_loader = ValidationDataset(images_info=train_images_info)
    # query_dataset = ValidationDataset(images_info=validation_images_info)
    # query_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False,
    #                                num_workers=args.num_workers)

    # dim = list(base_model.parameters())[-1].shape[0]  # last channels (512)

    # Define model for embedding


    descriptor_model = DgcnnModel(k=5, feature_dims=[64, 128], emb_dims=[256, 128], output_classes=args.descriptor_dim)
    net_vlad = NetVLAD(num_clusters=args.num_clusters, dim=args.descriptor_dim, alpha=1.0)

    # base_model_checkpoint = torch.load(os.path.join(args.saved_model_path, 'base_model.pth.tar'),
    #                                    map_location=lambda storage, loc: storage)
    # net_vlad_checkpoint = torch.load(os.path.join(args.saved_model_path, 'net_vlad.pth.tar'),
    #                                  map_location=lambda storage, loc: storage)
    # base_model.load_state_dict(base_model_checkpoint)
    if args.load_checkpoints:
        descriptor_model_checkpoint = torch.load(os.path.join(args.saved_model_path, 'descriptor_model.pth.tar'),
                                                 map_location=lambda storage, loc: storage)
        descriptor_model.load_state_dict(descriptor_model_checkpoint)
        net_vlad_checkpoint = torch.load(os.path.join(args.saved_model_path, 'net_vlad.pth.tar'),
                                                      map_location=lambda storage, loc: storage)
        net_vlad.load_state_dict(net_vlad_checkpoint)
        print("Loaded model checkpoints from \'{}\'.".format(os.path.join(args.saved_model_path, 'descriptor_model.pth.tar')))
        print("Loaded model checkpoints from \'{}\'.".format(
            os.path.join(args.saved_model_path, 'net_vlad.pth.tar')))

    optimizer = optim.Adam([
        {'params': descriptor_model.parameters(), 'lr': args.learning_rate},
        {'params': net_vlad.parameters(), 'lr': args.learning_rate}
    ])

    # optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # writer = SummaryWriter(log_dir=os.path.join(args.log_path, datetime.now().strftime('%b%d_%H-%M-%S')))

    ######################
    # draft for training #
    ######################
    descriptor_model.train()
    net_vlad.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    descriptor_model.to(device)
    net_vlad.to(device)
    criterion = nn.TripletMarginLoss(margin=args.margin ** 0.5, p=2, reduction='sum')

    # criterion = ContrastiveLoss(margin=2.0)
    with tqdm(train_data_loader) as tq:
        for query, positives, negatives in tq:

            for positive, negative in zip(positives, negatives):
                optimizer.zero_grad()
                query_vlad = net_vlad(
                    torch.cat([descriptor_model(segment.to(device)) for segment in query['segments']]))
                positive_vlad = net_vlad(
                    torch.cat([descriptor_model(segment.to(device)) for segment in positive['segments']]))
                negative_vlad = net_vlad(
                    torch.cat([descriptor_model(segment.to(device)) for segment in negative['segments']]))
                # print(query_vlad, positive_vlad, negative_vlad)
                loss = criterion(query_vlad.view(1, -1),positive_vlad.view(1, -1),negative_vlad.view(1, -1))
                print("loss: {}".format(loss.item()))
                loss.backward()

                del query_vlad, positive_vlad, negative_vlad


            # for positive, negative in zip(positives, negatives):
            #     # one step for positive
            #     optimizer.zero_grad()
            #     query_vlad = net_vlad(
            #         torch.cat([descriptor_model(segment.to(device)) for segment in query['segments']]))
            #     query_positive = net_vlad(
            #         torch.cat([descriptor_model(segment.to(device)) for segment in positive['segments']]))
            #     loss = criterion(query_vlad.view(1,-1), query_positive.view(1,-1), is_negative=False)
            #     loss.backward()
            #     del query_vlad, query_positive
            #     print("loss: {}".format(loss.item()))
            #
            #     # one step for negative
            #     optimizer.zero_grad()
            #     query_vlad = net_vlad(
            #         torch.cat([descriptor_model(segment.to(device)) for segment in query['segments']]))
            #     query_negative = net_vlad(
            #         torch.cat([descriptor_model(segment.to(device)) for segment in negative['segments']]))
            #     loss = criterion(query_vlad.view(1,-1), query_negative.view(1,-1), is_negative=True)
            #     loss.backward()
            #     del query_vlad, query_negative
            #     print("loss: {}".format(loss.item()))


            # descriptors_query = torch.cat([descriptor_model(segment.to(device)) for segment in query['segments']])



    # start training
    # for epoch in range(args.epochs):
    #     train(epoch, model, optimizer, train_data_loader, writer=writer)
    #     validate(model, train_images_info, validation_images_info, writer=None)


# TODO
def train(epoch, model, optimizer, train_data_loader, writer=None):
    print("Processing epoch {} ......".format(epoch))
    epoch_loss = 0
    iteration = 0
    model.train()
    device = torch.device("cuda" if args.use_gpu else "cpu")
    model.to(device)
    criterion = nn.TripletMarginLoss(margin=args.margin ** 0.5, p=2, reduction='sum')

    with tqdm(train_data_loader) as tq:
        for query, positives, negatives in tq:
            iteration += 1
            optimizer.zero_grad()

            descriptors_query = [model(segment.to(device)) for segment in query['segments']]
            assert len(positives) == len(negatives)
            for positive, negative in zip(positives,negatives):
                descriptors_positive = [model(segment.to(device)) for segment in positive['segments']]
                descriptors_negative = [model(segment.to(device)) for segment in negative['segments']]



            B, C, W, H = query.shape
            _, npos, _, _, _ = positives.shape  # B, npos, C, W, H
            _, nneg, _, _, _ = positives.shape  # B, nneg, C, W, H
            assert npos == nneg

            # print(query.shape)  # B * C * W * H
            # print(positives.shape)  # B * npos * C * W * H
            # print(negatives.shape)  # B * nneg * C * W * H
            input = torch.cat([query, positives.reshape(-1, C, W, H), negatives.reshape(-1, C, W, H)],
                              dim=0)  # (B * (1 + npos + nneg)) * C * W * H
            # input = input.reshape(-1, C, W, H) # (B * (1 + npos + nneg)) * C * W * H
            input = input.to(device)
            vlad_encoding = model(input)  # (B * (1 + npos + nneg))(17) * dim_vlad(16384)

            vladQ, vladP, vladN = torch.split(vlad_encoding, [B, B * npos, B * nneg])

            loss = 0
            for i_batch in range(B):
                index_offet = i_batch * npos
                for i in range(npos):
                    loss += criterion(vladQ[i_batch:i_batch + 1],
                                      vladP[index_offet + i:index_offet + i + 1],
                                      vladN[index_offet + i:index_offet + i + 1])
            loss /= B * npos
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
            del query, positives, negatives
    torch.cuda.empty_cache()
    # save model for each epoch
    torch.save(model.state_dict(), os.path.join(args.saved_model_path, 'model.pth.tar'))
    print("Saved models in \'{}\'.".format(os.path.join(args.saved_model_path, 'model.pth.tar')))


# TODO
def validate(model, database_images_info, query_images_info, writer=None):
    images_dir = os.path.join(args.dataset_dir, 'images')
    image_database = ImageDatabase(images_info=database_images_info,
                                   images_dir=images_dir, model=model,
                                   generate_database=True,
                                   transforms=input_transforms())
    true_count = 0
    for query_image_info in tqdm(query_images_info):
        query_results = image_database.query_image(
            image_filename=os.path.join(images_dir, query_image_info['image_file']), num_results=2)
        # print('query_result: \n{}'.format(query_results))
        for query_result in query_results:
            diff = query_image_info['position'] - query_result['position']
            if np.sqrt(diff @ diff) < args.positive_search_radius:
                true_count += 1
                break
    print("Precision: {}".format(true_count / len(query_images_info)))


if __name__ == '__main__':
    main()
    # train_demo()
