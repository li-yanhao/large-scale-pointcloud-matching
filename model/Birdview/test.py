import argparse
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from PIL import Image
from torchvision.models import resnet18

from dataset import *
from netvlad import EmbedNet
from netvlad import NetVLAD

parser = argparse.ArgumentParser(description='WayzNetVlad')
# parser.add_argument('--mode', type=str, default='train', help='Mode', choices=['train', 'validation'])
# parser.add_argument('--batch_size', type=int, default=2, help='batch_size')
parser.add_argument('--dataset_dir', type=str, default='/media/admini/My_data/0921/dataset_cam4', help='dataset_dir')
parser.add_argument('--num_workers', type=int, default=1, help='num_workers')
# parser.add_argument('--from_scratch', type=bool, default=True, help='from_scratch')
# parser.add_argument('--pretrained_embedding', type=bool, default=True, help='pretrained_embedding')
# parser.add_argument('--num_similar_neg', type=int, default=2, help='num_similar_neg')
# parser.add_argument('--margin', type=float, default=1.0, help='margin')
parser.add_argument('--use_gpu', type=bool, default=True, help='use_gpu')
# parser.add_argument('--learning_rate', type=float, default=0.0005, help='learning_rate')
# parser.add_argument('--positive_radius', type=float, default=0.3, help='positive_search_radius')
# parser.add_argument('--negative_filter_radius', type=float, default=2.0, help='negative_filter_radius')
parser.add_argument('--saved_model_path', type=str, default='saved_model', help='saved_model')
# parser.add_argument('--epochs', type=int, default=10, help='epochs')
parser.add_argument('--load_checkpoints', type=bool, default=True, help='load_checkpoints')
parser.add_argument('--num_clusters', type=int, default=64, help='num_clusters')
# parser.add_argument('--images_dir', type=str, default='/media/admini/My_data/0921/dataset_cam4/images',
#                     help='images_dir')
parser.add_argument('--queried_image', type=str, default='/media/admini/My_data/0921/dataset_cam4/images/001041.png',
                    help='queried_image')
parser.add_argument('--generate_database', type=bool, default=False, help='generate_database')

args = parser.parse_args()


def test():
    encoder = resnet18()
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

    # Define model for embedding
    net_vlad = NetVLAD(num_clusters=args.num_clusters, dim=dim, alpha=1.0)
    model = EmbedNet(base_model, net_vlad)

    # base_model_checkpoint = torch.load(os.path.join(args.saved_model_path, 'base_model.pth.tar'),
    #                                    map_location=lambda storage, loc: storage)
    # net_vlad_checkpoint = torch.load(os.path.join(args.saved_model_path, 'net_vlad.pth.tar'),
    #                                  map_location=lambda storage, loc: storage)
    # base_model.load_state_dict(base_model_checkpoint)
    # net_vlad.load_state_dict(net_vlad_checkpoint)
    model_checkpoint = torch.load(os.path.join(args.saved_model_path, 'model.pth.tar'),
                                  map_location=lambda storage, loc: storage)

    model.load_state_dict(model_checkpoint)
    print("Loaded model checkpoints from \'{}\'.".format(args.saved_model_path))

    # # torch.save(model.state_dict(), os.path.join(args.saved_model_path, 'model.pth.tar'))
    #
    # image_filenames = [os.path.join(args.images_dir, x) for x in os.listdir(args.images_dir) if
    #                    ImageDatabase.is_image_file(x)]

    images_info = make_images_info(args.dataset_dir, with_struct_file=False)
    images_dir = os.path.join(args.dataset_dir, 'images')
    if args.generate_database:
        image_database = ImageDatabase(images_info=images_info, images_dir=images_dir, model=model, generate_database=True)
        image_database.export_database('database_0921_cam4.npy')
    else:
        image_database = ImageDatabase(images_info=images_info, images_dir=images_dir, model=model,
                                       generate_database=False)
        image_database.import_database('database_0921_cam4.npy')

        queried_image_filename = args.queried_image
        queried_image_filenames = [
            queried_image_filename
            # '/media/admini/My_data/0923/dataset_xiaomi/images/1322.png',
            # '/media/admini/My_data/0923/dataset_xiaomi/images/1433.png',
            # '/media/admini/My_data/0923/dataset_xiaomi/images/1544.png',
            # '/media/admini/My_data/0923/dataset_xiaomi/images/1655.png',
            # '/media/admini/My_data/0923/dataset_xiaomi/images/1766.png',
            # '/media/admini/My_data/0923/dataset_xiaomi/images/1877.png',
        ]
        plotted_images = []
        num_results = 5
        for queried_image_filename in queried_image_filenames:
            query_results = image_database.query_image(queried_image_filename, num_results=num_results)
            # print('query_result: \n{}'.format(query_results))

            queried_image = Image.open(queried_image_filename)
            # result_image = Image.open(os.path.join(images_dir, query_results[0]['image_file']))
            result_images = [Image.open(os.path.join(images_dir, result['image_file'])) for result in query_results]

            plotted_images += [queried_image] + result_images
        plot_images(plotted_images, num_results+1)


def plot_images(images, cols):
    plt.figure()
    # plt.subplot(1, 1, 1)
    # plt.imshow(images[0])

    for i in range(0, len(images)):
        # row = i // cols
        # col = i % cols
        plt.subplot(len(images) // cols, cols, i + 1)
        plt.imshow(images[i])
    plt.show()


if __name__ == '__main__':
    test()
