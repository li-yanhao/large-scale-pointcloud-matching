import argparse
import os
from model.Birdview.dataset import make_images_info
from model.Birdview.dataset import NetVladDataset
from model.Birdview.dataset import PureDataset
from model.Birdview.base_model import BaseModel
from sklearn.model_selection import train_test_split
import torch
from model.Birdview.netvlad import NetVLAD, EmbedNet
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve
from matplotlib import pyplot as plt
import scipy.io as scio
import faiss


parser = argparse.ArgumentParser(description='metrics')
parser.add_argument('--dataset_dir', type=str, default='/media/admini/lavie/dataset/birdview_dataset/', help='dataset_dir')
parser.add_argument('--sequence_validate', type=str, default='08', help='sequence_validate')

# parser.add_argument('--dataset_dir', type=str, default='/home/li/Documents/wayz/image_data/dataset', help='dataset_dir')
parser.add_argument('--num_workers', type=int, default=1, help='num_workers')
parser.add_argument('--use_gpu', type=bool, default=True, help='use_gpu')
parser.add_argument('--positive_search_radius', type=float, default=5, help='positive_search_radius')
parser.add_argument('--negative_filter_radius', type=float, default=50, help='negative_filter_radius')
parser.add_argument('--saved_model_path', type=str,
                    default='/media/admini/lavie/dataset/birdview_dataset/saved_models', help='saved_model_path')
parser.add_argument('--num_clusters', type=int, default=64, help='num_clusters')
parser.add_argument('--final_dim', type=int, default=256, help='final_dim')
args = parser.parse_args()


def spi_vlad_roc_auc():
    images_info_validate = make_images_info(
        struct_filename=os.path.join(args.dataset_dir, 'struct_file_' + args.sequence_validate + '.txt'))

    validate_images_dir = os.path.join(args.dataset_dir, args.sequence_validate)

    # validate_database_images_info, validate_query_images_info = train_test_split(images_info_validate,
    #                                                                              test_size=0.4, random_state=20)

    base_model = BaseModel(300, 300)
    dim = 256

    # Define model for embedding
    net_vlad = NetVLAD(num_clusters=args.num_clusters, dim=dim, alpha=1.0, outdim=args.final_dim)
    model = EmbedNet(base_model, net_vlad)

    saved_model_file = os.path.join(args.saved_model_path, 'model-to-check-top1.pth.tar')

    model_checkpoint = torch.load(saved_model_file, map_location=lambda storage, loc: storage)
    model.load_state_dict(model_checkpoint)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(dev)
    print("Loaded model checkpoints from \'{}\'.".format(saved_model_file))

    validate_dataset = PureDataset(images_info=images_info_validate, images_dir=validate_images_dir)
    validate_data_loader = DataLoader(validate_dataset, batch_size=1)

    descriptors = []
    positions = []
    torch.set_grad_enabled(False)
    i = 0
    for query, query_info in tqdm(validate_data_loader):
        # print(query.shape)
        netvlad_encoding = model(query.to(dev)).cpu().view(-1)
        # print(netvlad_encoding.shape)
        descriptors.append(netvlad_encoding)
        position = query_info['position'].view(3)
        positions.append(position)
        i = i + 1
        # if i > 100:
        #     break
        # print(netvlad_encoding)
    descriptors = torch.cat(descriptors).view(-1, args.final_dim)
    # print(descriptors.shape)

    N = len(descriptors)
    diff = descriptors[...,None] - descriptors.transpose(0,1)[None,...]
    score_matrix = (1 - torch.einsum('mdn,mdn->mn', diff, diff)).numpy()
    # print(score_matrix)

    positions = torch.cat(positions).view(-1, 3)
    diff = positions[..., None] - positions.transpose(0, 1)[None, ...]
    label_matrix = (torch.einsum('mdn,mdn->mn', diff, diff) < (args.positive_search_radius**2)).numpy()
    # print(label_matrix.reshape(-1))
    # print(score_matrix.reshape(-1))

    print('AUC:', roc_auc_score(label_matrix.reshape(-1), score_matrix.reshape(-1)))

    # print('F1-score:', f1_score(label_matrix.reshape(-1), score_matrix.reshape(-1)))

    precision, recall, thresholds = precision_recall_curve(label_matrix.reshape(-1), score_matrix.reshape(-1))
    print(recall, precision)
    plt.plot(recall, precision, lw=1)

    # plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label="Luck")
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel("Recall Rate")
    plt.ylabel("Precision Rate")
    plt.show()

    descriptors = descriptors.numpy()
    np.save('spi-vlad-kitti'+args.sequence_validate+'.npy', descriptors)


    torch.set_grad_enabled(True)
    # label_matrix =
    pass


def m2dp_auc():
    struct_filename = '/media/admini/LENOVO/dataset/kitti/lidar_odometry/birdview_dataset/tmp/struct_file_one_scan_05.txt'
    images_info = make_images_info(struct_filename=struct_filename)
    print(len(images_info))

    positions = np.array([image_info['position'] for image_info in images_info])
    print(positions.shape)

    data_file = '/home/admini/yanhao/large-scale-pointcloud-matching/m2dp-kitti05.mat'
    descriptors = scio.loadmat(data_file)['descriptors'][:-1]
    print(descriptors.shape)

    descriptors = torch.Tensor(descriptors)
    positions = torch.Tensor(positions)
    diff = descriptors[..., None] - descriptors.transpose(0, 1)[None, ...]
    score_matrix = (1 - torch.einsum('mdn,mdn->mn', diff, diff)).numpy()

    diff = positions[..., None] - positions.transpose(0, 1)[None, ...]
    label_matrix = (torch.einsum('mdn,mdn->mn', diff, diff) < (args.positive_search_radius ** 2)).numpy()

    print('AUC:', roc_auc_score(label_matrix.reshape(-1), score_matrix.reshape(-1)))
    pass


def sc_ringkeys_auc():
    struct_filename = '/media/admini/LENOVO/dataset/kitti/lidar_odometry/birdview_dataset/tmp/struct_file_one_scan_08.txt'
    images_info = make_images_info(struct_filename=struct_filename)
    print(len(images_info))

    positions = np.array([image_info['position'] for image_info in images_info])
    print(positions.shape)

    data_file = '/home/admini/yanhao/large-scale-pointcloud-matching/sc-ringkeys-kitti08.mat'
    descriptors = scio.loadmat(data_file)['ringkeys'][1:]
    print(descriptors.shape)

    descriptors = torch.Tensor(descriptors)
    positions = torch.Tensor(positions)
    diff = descriptors[..., None] - descriptors.transpose(0, 1)[None, ...]
    score_matrix = (1 - torch.einsum('mdn,mdn->mn', diff, diff)).numpy()

    diff = positions[..., None] - positions.transpose(0, 1)[None, ...]
    label_matrix = (torch.einsum('mdn,mdn->mn', diff, diff) < (args.positive_search_radius ** 2)).numpy()

    print('AUC:', roc_auc_score(label_matrix.reshape(-1), score_matrix.reshape(-1)))
    pass



def sc_ringkeys_auc():
    struct_filename = '/media/admini/LENOVO/dataset/kitti/lidar_odometry/birdview_dataset/tmp/struct_file_05_ds.txt'
    images_info = make_images_info(struct_filename=struct_filename)
    print(len(images_info))

    positions = np.array([image_info['position'] for image_info in images_info])
    print(positions.shape)

    data_file = '/home/admini/yanhao/large-scale-pointcloud-matching/pnv-kitti05.npy'
    descriptors = np.load(data_file)
    print(descriptors.shape)

    descriptors = torch.Tensor(descriptors)
    positions = torch.Tensor(positions)
    diff = descriptors[..., None] - descriptors.transpose(0, 1)[None, ...]
    score_matrix = (1 - torch.einsum('mdn,mdn->mn', diff, diff)).numpy()

    diff = positions[..., None] - positions.transpose(0, 1)[None, ...]
    label_matrix = (torch.einsum('mdn,mdn->mn', diff, diff) < (args.positive_search_radius ** 2)).numpy()

    print('AUC:', roc_auc_score(label_matrix.reshape(-1), score_matrix.reshape(-1)))
    pass


def top_k_m2dp():
    k = 25
    positive_radius = 3
    struct_filename = '/media/admini/LENOVO/dataset/kitti/lidar_odometry/birdview_dataset/tmp/struct_file_one_scan_02.txt'
    images_info = make_images_info(struct_filename=struct_filename)
    print(len(images_info))

    positions = np.array([image_info['position'] for image_info in images_info])
    print(positions.shape)

    data_file = '/home/admini/yanhao/large-scale-pointcloud-matching/m2dp-kitti02.mat'
    descriptors = scio.loadmat(data_file)['descriptors'][:-1]
    print(descriptors.shape)

    descriptors = np.asarray(descriptors, order='C').astype('float32')

    database_descriptors, query_descriptors, database_positions, query_positions = \
        train_test_split(descriptors, positions, test_size=0.4, random_state=10)

    index = faiss.IndexFlatL2(database_descriptors.shape[-1])
    index.add(database_descriptors)
    topk_score_overall = np.zeros(k)
    for descriptor, position in zip(query_descriptors, query_positions):
        distances, indices = index.search(descriptor.reshape(1,-1), k)
        candidate_positions = database_positions[indices[0]]
        diff = candidate_positions - position
        # print((diff * diff).sum(axis=1))
        # print(diff)
        is_true_result = (diff * diff).sum(axis=1) < positive_radius**2
        topk_score = is_true_result.cumsum() > 0
        # print(topk_score)
        topk_score_overall += topk_score
    topk_score_overall /= len(query_descriptors)
    print(topk_score_overall)
    print(database_descriptors.shape)
    print(query_descriptors.shape)
    print(database_positions.shape)
    print(query_positions.shape)


    pass


if __name__ == '__main__':
    # spi_vlad_roc_auc()
    # m2dp_auc()
    # sc_ringkeys_auc()
    # sc_ringkeys_auc()
    top_k_m2dp()