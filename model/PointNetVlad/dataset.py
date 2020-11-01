import os

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import torchvision.transforms as transforms
from sklearn.neighbors import NearestNeighbors
import faiss
from tqdm import tqdm
import open3d as o3d
from scipy.spatial.transform import Rotation as R


def make_ptclouds_info(struct_filename):
    ptclouds_info = []
    if struct_filename is not None:
        with open(struct_filename, "r") as struct_file:
            # skip the first line
            struct_file.readline()
            while True:
                line = struct_file.readline()
                if not line:
                    break
                split = [i for i in line.split(",")]
                ptclouds_info.append({
                    'pcd_file': split[0][:-4]+'.pcd',
                    'timestamp': float(split[1]),
                    'position': np.array([float(split[2]), float(split[3]), float(split[4])]),
                    # w, x, y, z
                    'orientation': np.array(
                        [float(split[5]), float(split[6]), float(split[7]), float(split[8])])
                })
    return np.array(ptclouds_info)


class PNVDataset(Dataset):
    def __init__(self, ptclouds_info, ptclouds_dir, positive_search_radius=0.5,
                 negative_filter_radius=2.0, num_similar_negatives=8, num_points=4096, print_query_info=False,
                 add_rotation=True):
        super(PNVDataset, self).__init__()
        self.ptclouds_info = ptclouds_info
        self.ptclouds_dir = ptclouds_dir
        self.for_database = False
        self.ptclouds_info = np.array(self.ptclouds_info)
        self.print_query_info = print_query_info
        self.num_points = num_points
        self.add_rotation = add_rotation

        self._generate_train_dataset(positive_search_radius, negative_filter_radius, num_similar_negatives)

    # TODO: Determine (query, positive, negative) indices before feeding data
    def _generate_train_dataset(self, positive_search_radius, negative_filter_radius, num_similar_negatives):
        knn = NearestNeighbors()
        image_positions = np.array([image_info['position'] for image_info in self.ptclouds_info])
        knn.fit(image_positions)
        list_of_distances, self.list_of_tmp_positives_indices = knn.radius_neighbors(image_positions,
                                                                                     radius=positive_search_radius,
                                                                                     sort_results=True)
        self.list_of_positives_indices = []

        query_index = 0
        # max_angle_diff_in_radian = max_angle_diff_in_degree / 180 * np.pi
        i = 0
        for distances, tmp_positive_indices in zip(list_of_distances, self.list_of_tmp_positives_indices):
            assert len(tmp_positive_indices) > 1
            # print(self.ptclouds_info[i]["image_file"])
            i += 1
            tmp_positive_indices = tmp_positive_indices[1:]  # indices[0] is the query sample, remove it

            positive_indices = tmp_positive_indices
            assert len(positive_indices) > 0
            # probabilities = np.array(probabilities)
            # probabilities /= probabilities.sum()
            # self.list_of_positives_indices.append(
            #     np.random.choice(positive_indices, 1, replace=True))
            self.list_of_positives_indices.append(positive_indices)
            query_index += 1
        # self.list_of_positives_indices = [None if len(indices)<1 else indices[1:] for indices in self.list_of_positives_indices]

        # print(self.list_of_positives_indices)

        self.list_of_negative_indices = []
        self.list_of_unrelated_indices = []
        for i in range(len(self.ptclouds_info)):
            _, non_negative_indices = knn.radius_neighbors(image_positions[i].reshape(1, -1),
                                                           radius=negative_filter_radius)
            non_negative_indices = non_negative_indices[0]
            negative_indices = np.setdiff1d(np.arange(len(self.ptclouds_info)), non_negative_indices, assume_unique=True)
            knn_negatives = NearestNeighbors()
            knn_negatives.fit(image_positions[negative_indices])
            _, nn_indices = knn_negatives.kneighbors(image_positions[i].reshape(1, -1), num_similar_negatives)
            similar_negative_indices = negative_indices[nn_indices[0]]
            random_negative_indices = np.setdiff1d(negative_indices, similar_negative_indices, assume_unique=True)
            random_negative_indices = np.random.choice(random_negative_indices, num_similar_negatives,
                                                       replace=False)
            # print(similar_negative_indices)
            # print(random_negative_indices)
            merged_negative_indices = np.concatenate([similar_negative_indices, random_negative_indices])
            self.list_of_negative_indices.append(merged_negative_indices)

            unrelated_indices = np.random.choice(non_negative_indices, 30, replace=True)
            self.list_of_unrelated_indices.append(unrelated_indices)

    @staticmethod
    def _rotate(points, angle_in_radius):
        """
        points: 1 * N * 3
        """
        rotation_matrix = torch.Tensor(
            R.from_rotvec(angle_in_radius * np.array([0, 0, 1])).as_matrix())
        return points @ torch.Tensor(rotation_matrix)

    def __len__(self):
        return len(self.ptclouds_info)

    def __getitem__(self, index):
        query_pcd = o3d.io.read_point_cloud(os.path.join(self.ptclouds_dir, self.ptclouds_info[index]['pcd_file']))
        query = torch.Tensor(query_pcd.points)
        pt_entries = np.random.choice(len(query), self.num_points, replace=False)
        query = query[pt_entries]
        # query_pcd.points = o3d.utility.Vector3dVector(query.numpy())
        # o3d.visualization.draw_geometries([query_pcd])
        if self.add_rotation:
            query = self._rotate(query, np.random.ranf() * 2 * np.pi - np.pi)
        query = query[None,...]

        pos_indices = self.list_of_positives_indices[index]
        pos_index = np.random.choice(pos_indices, 1, replace=True)[0]
        positive_pcd = o3d.io.read_point_cloud(os.path.join(self.ptclouds_dir, self.ptclouds_info[pos_index]['pcd_file']))
        positive = torch.Tensor(positive_pcd.points)
        pt_entries = np.random.choice(len(positive), self.num_points, replace=False)
        positive = positive[pt_entries]
        # positive_pcd.points = o3d.utility.Vector3dVector(positive.numpy())
        # o3d.visualization.draw_geometries([positive_pcd])
        if self.add_rotation:
            positive = self._rotate(positive, np.random.ranf() * 2 * np.pi - np.pi)
        positive = positive[None, ...]

        negatives = []
        for neg_index in self.list_of_negative_indices[index]:
            negative_pcd = o3d.io.read_point_cloud(
                os.path.join(self.ptclouds_dir, self.ptclouds_info[neg_index]['pcd_file']))
            negative = torch.Tensor(negative_pcd.points)
            pt_entries = np.random.choice(len(negative), self.num_points, replace=False)
            negative = negative[pt_entries]
            if self.add_rotation:
                negative = self._rotate(negative, np.random.ranf() * 2 * np.pi - np.pi)
            negative = negative[None, None, ...]
            negatives.append(negative)
        negatives = torch.cat(negatives)

        unrelated_index = np.random.choice(self.list_of_unrelated_indices[index], 1)[0]
        unrelated_pcd = o3d.io.read_point_cloud(
            os.path.join(self.ptclouds_dir, self.ptclouds_info[unrelated_index]['pcd_file']))
        unrelated = torch.Tensor(unrelated_pcd.points)
        pt_entries = np.random.choice(len(unrelated), self.num_points, replace=False)
        unrelated = unrelated[pt_entries]
        if self.add_rotation:
            unrelated = self._rotate(unrelated, np.random.ranf() * 2 * np.pi - np.pi)
        unrelated = unrelated[None, ...]

        if self.print_query_info:
            return query, positive, negatives, self.ptclouds_info[index]

        return query, positive, negatives, unrelated


# for ptcloud retrieval
class PNVDatabase(object):
    def __init__(self, ptclouds_info: list, ptclouds_dir: str, model, num_points, generate_database=False):
        self.model = model
        self.input_transforms = transforms
        self.database = None
        self.index = None
        self.ptclouds_info = ptclouds_info.copy()
        self.ptclouds_dir = ptclouds_dir
        self.num_points = num_points

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        self.descriptors = []

        if generate_database:
            self._generate_database()

    @torch.no_grad()
    def _generate_database(self):
        assert len(self.ptclouds_info) > 0
        encodings = []
        # self.database = []
        print('Generating database from \'{}\'...'.format(self.ptclouds_dir))
        with torch.no_grad():
            for ptcloud_info in tqdm(self.ptclouds_info):
                input_pcd = o3d.io.read_point_cloud(os.path.join(self.ptclouds_dir, ptcloud_info['pcd_file']))
                input = torch.Tensor(input_pcd.points)
                pt_entries = np.random.choice(len(input), self.num_points, replace=False)
                input = input[pt_entries]
                # query_pcd.points = o3d.utility.Vector3dVector(query.numpy())
                # o3d.visualization.draw_geometries([query_pcd])
                input = input[None, None, ...].to(self.device)
                netvlad_encoding = self.model(input).cpu().numpy().squeeze()
                ptcloud_info['encoding'] = netvlad_encoding
                encodings.append(netvlad_encoding)

        dim_encoding = len(encodings[0])
        encodings = np.array(encodings)
        np.save('pnv.npy', encodings)
        self.index = faiss.IndexFlatL2(dim_encoding)
        self.index.add(encodings)
        # self.model.cpu()
        self.ptclouds_info = np.array(self.ptclouds_info)
        print("Generation of database finished")

    def export_database(self, filename):
        np.save(filename, self.ptclouds_info)
        print('Exported database to {}'.format(filename))

    def import_database(self, filename):
        self.ptclouds_info = np.load(filename, allow_pickle=True)
        encodings = [datum['encoding'] for datum in self.ptclouds_info]
        dim_encoding = len(encodings[0])
        encodings = np.array(encodings)
        self.index = faiss.IndexFlatL2(dim_encoding)
        self.index.add(encodings)
        print('Imported database from {}'.format(filename))

    @torch.no_grad()
    def query_ptcloud(self, ptcloud_filename, num_results=1):
        assert len(self.ptclouds_info) > 0
        input_pcd = o3d.io.read_point_cloud(ptcloud_filename)

        input = torch.Tensor(input_pcd.points)
        pt_entries = np.random.choice(len(input), self.num_points, replace=False)
        input = input[pt_entries]
        # query_pcd.points = o3d.utility.Vector3dVector(query.numpy())
        # o3d.visualization.draw_geometries([query_pcd])
        input = input[None, None, ...].to(self.device)

        netvlad_encoding = self.model(input).cpu().numpy()
        # netvlad_encoding = netvlad_encoding.unsqueeze(0)
        distances, indices = self.index.search(netvlad_encoding, num_results)
        return self.ptclouds_info[indices[0]]


if __name__ == '__main__':
    # dataset_dir = '/media/admini/My_data/0904/dataset'
    ptclouds_info = make_ptclouds_info(
        struct_filename='/media/admini/LENOVO/dataset/kitti/lidar_odometry/birdview_dataset/tmp/struct_file_00.txt')
    dataset_dir = '/media/admini/LENOVO/dataset/kitti/lidar_odometry/birdview_dataset/tmp'
    ptclouds_dir = '/media/admini/LENOVO/dataset/kitti/lidar_odometry/birdview_dataset/tmp/00'

    dataset = PNVDataset(ptclouds_info=ptclouds_info, ptclouds_dir=ptclouds_dir, positive_search_radius=8,
                 negative_filter_radius=50, num_similar_negatives=4)
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True)
    # data_loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn, shuffle=True)
    # with tqdm(data_loader) as tq:
    i = 0
    for query, positive, negatives, unrelated in data_loader:
        # print(item)
        print(negatives.shape)
        print(i)
        i += 1
    print(len(dataset))
