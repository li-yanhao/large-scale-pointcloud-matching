import os
import argparse
from model.Birdview.dataset import *
from model.SapientNet.superpoint import *
from torch.utils.data import DataLoader
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.autograd import Variable

from model.Birdview.netvlad import NetVLAD
from model.Birdview.netvlad import EmbedNet
from model.Birdview.loss import HardTripletLoss
from model.Birdview.base_model import BaseModel
from torchvision.models import resnet18, vgg16
from PIL import Image
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from datetime import datetime
import numpy as np
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(description='WayzNetVlad')
parser.add_argument('--mode', type=str, default='train', help='Mode', choices=['train', 'test'])
parser.add_argument('--batch_size', type=int, default=1, help='batch_size')
parser.add_argument('--dataset_dir', type=str, default='/media/admini/lavie/dataset/birdview_dataset/', help='dataset_dir')
parser.add_argument('--sequence_train', type=str, default='00', help='sequence_train')
parser.add_argument('--sequence_validate', type=str, default='08', help='sequence_validate')

# parser.add_argument('--dataset_dir', type=str, default='/home/li/Documents/wayz/image_data/dataset', help='dataset_dir')
parser.add_argument('--num_workers', type=int, default=1, help='num_workers')
# parser.add_argument('--from_scratch', type=bool, default=True, help='from_scratch')
parser.add_argument('--pretrained_embedding', type=bool, default=False, help='pretrained_embedding')
parser.add_argument('--num_similar_neg', type=int, default=2, help='number of similar negative samples')
parser.add_argument('--margin', type=float, default=0.5, help='margin')
parser.add_argument('--use_gpu', type=bool, default=True, help='use_gpu')
parser.add_argument('--learning_rate', type=float, default=0.0005, help='learning_rate')
parser.add_argument('--positive_search_radius', type=float, default=8, help='positive_search_radius')
parser.add_argument('--negative_filter_radius', type=float, default=50, help='negative_filter_radius')
parser.add_argument('--saved_model_path', type=str,
                    default='/media/admini/lavie/dataset/birdview_dataset/saved_models', help='saved_model_path')
parser.add_argument('--epochs', type=int, default=120, help='epochs')
parser.add_argument('--load_checkpoints', type=bool, default=False, help='load_checkpoints')
parser.add_argument('--num_clusters', type=int, default=64, help='num_clusters')
parser.add_argument('--final_dim', type=int, default=256, help='final_dim')
parser.add_argument('--log_path', type=str, default='logs', help='log_path')
args = parser.parse_args()


def train_demo():
    # Discard layers at the end of base network
    encoder = resnet18(pretrained=True)
    base_model = nn.Sequential(
        encoder.conv1,
        encoder.bn1,
        encoder.relu,
        encoder.maxpool,
        encoder.layer1,
        encoder.layer2,
        encoder.layer3,
        encoder.layer4,
    )


    dim = list(base_model.parameters())[-1].shape[0]  # last channels (512)

    # Define model for embedding
    net_vlad = NetVLAD(num_clusters=args.num_clusters, dim=dim, alpha=1.0)
    model = EmbedNet(base_model, net_vlad).cuda()

    # Define loss
    criterion = HardTripletLoss(margin=0.1).cuda()

    # This is just toy example. Typically, the number of samples in each classes are 4.
    labels = torch.randint(0, 10, (40,)).long()
    x = torch.rand(40, 3, 128, 128).cuda()
    output = model(x)  # 40 * 16384

    triplet_loss = criterion(output, labels)


def main():
    images_info_train = make_images_info(
        struct_filename=os.path.join(args.dataset_dir, 'struct_file_' + args.sequence_train + '.txt'))
    images_info_validate = make_images_info(
        struct_filename=os.path.join(args.dataset_dir, 'struct_file_' + args.sequence_validate + '.txt'))

    validate_images_dir = os.path.join(args.dataset_dir, args.sequence_validate)
    train_images_dir = os.path.join(args.dataset_dir, args.sequence_train)

    train_database_images_info, train_query_images_info = train_test_split(images_info_train, test_size=0.1, random_state=42)

    validate_database_images_info, validate_query_images_info = train_test_split(images_info_validate, test_size=0.2, random_state=20)
    train_dataset = NetVladDataset(images_info=train_database_images_info, images_dir=train_images_dir,
                                   num_similar_negatives=args.num_similar_neg,
                                   positive_search_radius=args.positive_search_radius,
                                   negative_filter_radius=args.negative_filter_radius)
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                   num_workers=args.num_workers)

    encoder = resnet18(pretrained=args.pretrained_embedding)
    base_model = nn.Sequential(
        encoder.conv1,
        encoder.bn1,
        encoder.relu,
        encoder.maxpool,
        encoder.layer1,
        encoder.layer2,
        encoder.layer3,
        encoder.layer4
    )
    dim = list(base_model.parameters())[-1].shape[0]  # last channels (512)

    base_model = BaseModel(300, 300)
    dim = 256

    # Define model for embedding
    net_vlad = NetVLAD(num_clusters=args.num_clusters, dim=dim, alpha=1.0, outdim=args.final_dim)
    model = EmbedNet(base_model, net_vlad)


    saved_model_file = os.path.join(args.saved_model_path, 'model-resnet18.pth.tar')
    if args.load_checkpoints:
        # base_model_checkpoint = torch.load(os.path.join(args.saved_model_path, 'base_model.pth.tar'),
        #                                    map_location=lambda storage, loc: storage)
        # net_vlad_checkpoint = torch.load(os.path.join(args.saved_model_path, 'net_vlad.pth.tar'),
        #                                  map_location=lambda storage, loc: storage)
        # base_model.load_state_dict(base_model_checkpoint)
        # net_vlad.load_state_dict(net_vlad_checkpoint)
        model_checkpoint = torch.load(saved_model_file, map_location=lambda storage, loc: storage)
        model.load_state_dict(model_checkpoint)
        print("Loaded model checkpoints from \'{}\'.".format(saved_model_file))

    # base_model.train()
    # net_vlad.train()
    model.train()

    # optimizer = optim.Adam([base_model.parameters(), net_vlad.parameters()], lr=args.learning_rate)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    writer = SummaryWriter(log_dir=os.path.join(args.log_path, datetime.now().strftime('%b%d_%H-%M-%S')))

    # start training
    # if args.mode == 'train':
    for epoch in range(args.epochs):
        # validate(model, train_images_info, validation_images_info, writer=None)
        epoch = epoch + 1

        if epoch % 1 == 0:
            validate(model, validate_database_images_info, validate_query_images_info, validate_images_dir, writer=None)
            validate(model, train_database_images_info, train_query_images_info, train_images_dir, writer=None)
        train(epoch, model, optimizer, train_data_loader, writer=None)
        torch.save(model.state_dict(), saved_model_file)
        print("Saved models in \'{}\'.".format(saved_model_file))


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
        for query, positives, negatives in tq:
            iteration += 1
            optimizer.zero_grad()

            B, C, W, H = query.shape
            _, npos, _, _, _ = positives.shape  # B, npos, C, W, H
            _, nneg, _, _, _ = negatives.shape  # B, nneg, C, W, H
            # assert npos == nneg

            # print(query.shape)  # B * C * W * H
            # print(positives.shape)  # B * npos * C * W * H
            # print(negatives.shape)  # B * nneg * C * W * H
            input = torch.cat([query, positives.reshape(-1, C, W, H), negatives.reshape(-1, C, W, H)],
                              dim=0)  # (B * (1 + npos + nneg)) * C * W * H
            # input = input.reshape(-1, C, W, H) # (B * (1 + npos + nneg)) * C * W * H
            input = input.to(device)
            vlad_encoding = model(input)  # (B * (1 + npos + nneg))(17) * dim_vlad(16384)
            # print(vlad_encoding.shape, B, npos, nneg)
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


# TODO
def validate(model, database_images_info, query_images_info, images_dir, writer=None):
    image_database = ImageDatabase(images_info=database_images_info,
                                   images_dir=images_dir, model=model,
                                   generate_database=True,
                                   transforms=input_transforms())
    true_count = 0
    for query_image_info in tqdm(query_images_info):
        query_results = image_database.query_image(
            image_filename=os.path.join(images_dir, query_image_info['image_file']), num_results=3)
        # print('query_result: \n{}'.format(query_results))
        for query_result in query_results:
            diff = query_image_info['position'] - query_result['position']
            if np.sqrt(diff @ diff) < args.positive_search_radius:
                true_count += 1
                break
    print("Precision: {}".format(true_count / len(query_images_info)))


def model_test():
    base_model = BaseModel(200, 200)
    dim = 256
    net_vlad = NetVLAD(num_clusters=args.num_clusters, dim=dim, alpha=1.0)
    model = EmbedNet(base_model, net_vlad)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epoch_loss = 0
    iteration = 0
    model.train()
    device = torch.device("cuda" if args.use_gpu else "cpu")
    # device = torch.device("cpu")
    model.to(device)
    criterion = nn.TripletMarginLoss(margin=args.margin ** 0.5, p=2, reduction='sum')

    for i in range(10):
        B, C, W, H = 2, 1, 200, 200
        query = torch.randn(B, C, W, H)
        positives = torch.randn(B, 2, C, W, H)
        negatives = torch.randn(B, 2, C, W, H)
        iteration += 1
        optimizer.zero_grad()

        B, C, W, H = query.shape
        _, npos, _, _, _ = positives.shape  # B, npos, C, W, H
        _, nneg, _, _, _ = negatives.shape  # B, nneg, C, W, H
        # assert npos == nneg

        # print(query.shape)  # B * C * W * H
        # print(positives.shape)  # B * npos * C * W * H
        # print(negatives.shape)  # B * nneg * C * W * H
        input = torch.cat([query, positives.reshape(-1, C, W, H), negatives.reshape(-1, C, W, H)],
                          dim=0)  # (B * (1 + npos + nneg)) * C * W * H
        # input = input.reshape(-1, C, W, H) # (B * (1 + npos + nneg)) * C * W * H
        input = input.to(device)
        vlad_encoding = model(input)  # (B * (1 + npos + nneg))(17) * dim_vlad(16384)
        # print(vlad_encoding.shape, B, npos, nneg)
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
        print("loss: ", loss.item())

        del input, vlad_encoding, vladQ, vladP, vladN
        del query, positives, negatives
    torch.cuda.empty_cache()




if __name__ == '__main__':
    main()
    # train_demo()
    # model_test()
