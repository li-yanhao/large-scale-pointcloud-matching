import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import torchvision.transforms as transforms
from sklearn.neighbors import NearestNeighbors
import faiss
from tqdm import tqdm
from scipy.spatial.transform import Rotation


def input_transforms():
    return transforms.Compose([
        transforms.Resize(size=(300, 300)),
        # transforms.RandomResizedCrop(size=(600, 960), scale=(0.8, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=2),
        transforms.RandomRotation(degrees=360),
        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor(),
        # transforms.RandomErasing(p=0.3, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                      std=[0.229, 0.224, 0.225]),
    ])


def input_transforms_test():
    return transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                      std=[0.229, 0.224, 0.225]),
    ])

def is_image_file(filename):
    return any(filename.lower().endswith(extension) for extension in ['.bmp', '.png', '.jpg', '.jpeg'])


def make_images_info(struct_filename=None, images_dir=None):
    assert(struct_filename is not None or images_dir is not None)
    images_info = []
    if struct_filename is not None:
        with open(struct_filename, "r") as struct_file:
            # skip the first line
            struct_file.readline()
            while True:
                line = struct_file.readline()
                if not line:
                    break
                split = [i for i in line.split(",")]
                images_info.append({
                    'image_file': split[0],
                    'timestamp': float(split[1]),
                    'position': np.array([float(split[2]), float(split[3]), float(split[4])]),
                    'orientation': np.array(
                        [float(split[5]), float(split[6]), float(split[7]), float(split[8])])
                })
    else:
        images_info = [{'image_file': image_filename} for image_filename in os.listdir(images_dir) if
                       is_image_file(image_filename)]
    return np.array(images_info)


class NetVladDataset(Dataset):
    def __init__(self, images_info, images_dir, transforms=input_transforms(), positive_search_radius=0.5,
                 negative_filter_radius=2.0, num_similar_negatives=8, max_angle_diff_in_degree=90):
        super(NetVladDataset, self).__init__()
        self.input_transforms = transforms
        self.images_info = images_info
        self.images_dir = images_dir

        self.for_database = False

        # with open(os.path.join(dataset_dir, "image_struct.txt"), "r") as struct_file:
        #     # skip the first line
        #     struct_file.readline()
        #     while True:
        #         line = struct_file.readline()
        #         if not line:
        #             break
        #         splitted = [i for i in line.split(",")]
        #         self.images_info.append({
        #             'image_file': splitted[0],
        #             'timestamp': float(splitted[1]),
        #             'position': np.array([float(splitted[2]), float(splitted[3]), float(splitted[4])]),
        #             'orientation': np.array(
        #                 [float(splitted[5]), float(splitted[6]), float(splitted[7]), float(splitted[8])])
        #         })
        self.images_info = np.array(self.images_info)
        self._generate_train_dataset(positive_search_radius, negative_filter_radius, num_similar_negatives,
                                     max_angle_diff_in_degree)

    # TODO: Determine (query, positive, negative) indices before feeding data
    def _generate_train_dataset(self, positive_search_radius, negative_filter_radius, num_similar_negatives,
                                max_angle_diff_in_degree):
        knn = NearestNeighbors()
        image_positions = np.array([image_info['position'] for image_info in self.images_info])
        knn.fit(image_positions)
        list_of_distances, self.list_of_tmp_positives_indices = knn.radius_neighbors(image_positions,
                                                                                     radius=positive_search_radius,
                                                                                     sort_results=True)
        self.list_of_positives_indices = []

        query_index = 0
        max_angle_diff_in_radian = max_angle_diff_in_degree / 180 * np.pi
        i = 0
        for distances, tmp_positive_indices in zip(list_of_distances, self.list_of_tmp_positives_indices):
            assert len(tmp_positive_indices) > 1
            # print(self.images_info[i]["image_file"])
            i += 1
            tmp_positive_indices = tmp_positive_indices[1:]  # indices[0] is the query sample, remove it
            # TODO: filter out the samples with large orientation difference
            # tmp_probabilities = np.exp(distances)
            # positive_indices = []
            # probabilities = []
            # for probability, tmp_index in zip(tmp_probabilities, tmp_positive_indices):
            #     query_orientation = Rotation.from_quat(self.images_info[query_index]['orientation'])
            #     positive_orientation = Rotation.from_quat(self.images_info[tmp_index]['orientation'])
                # angle = abs((query_orientation * positive_orientation.inv()).magnitude())
                # if angle < max_angle_diff_in_radian:
                #     positive_indices.append(tmp_index)
                #     probabilities.append(probability)
            positive_indices = tmp_positive_indices
            assert len(positive_indices) > 0
            # probabilities = np.array(probabilities)
            # probabilities /= probabilities.sum()
            self.list_of_positives_indices.append(
                np.random.choice(positive_indices, num_similar_negatives * 2, replace=True))
            query_index += 1
        self.list_of_positives_indices = [None if len(indices)<=1 else indices[1:] for indices in self.list_of_positives_indices]
        # print(self.list_of_positives_indices)

        self.list_of_negative_indices = []
        for i in range(len(self.images_info)):
            _, non_negative_indices = knn.radius_neighbors(image_positions[i].reshape(1, -1),
                                                           radius=negative_filter_radius)
            non_negative_indices = non_negative_indices[0]
            negative_indices = np.setdiff1d(np.arange(len(self.images_info)), non_negative_indices, assume_unique=True)
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

    def __len__(self):
        return len(self.images_info)

    def __getitem__(self, index):
        query = Image.open(os.path.join(self.images_dir, self.images_info[index]['image_file']))


        # Image.open('PATH').convert('RGB')

        positives = []
        for pos_index in self.list_of_positives_indices[index]:
            positives.append(Image.open(os.path.join(self.images_dir, self.images_info[pos_index]['image_file'])))
        negatives = []
        for neg_index in self.list_of_negative_indices[index]:
            negatives.append(Image.open(os.path.join(self.images_dir, self.images_info[neg_index]['image_file'])))

        if self.input_transforms:
            query = self.input_transforms(query)
            negatives = torch.cat([self.input_transforms(img).unsqueeze(0) for img in negatives])
            positives = torch.cat([self.input_transforms(img).unsqueeze(0) for img in positives])
        return query, positives, negatives


class ValidationDataset(Dataset):
    def __init__(self, images_info):
        self.images_info = images_info

    def __len__(self):
        return len(self.images_info)

    def __getitem__(self, index):
        query = Image.open(self.images_info[index]['image_file'])
        if self.input_transforms:
            query = self.input_transforms(query)
        return query


# class ValidationDatabase(object):
#     def


class PureDataset(Dataset):
    def __init__(self, dataset_dir: str, transforms=input_transforms()):
        super(PureDataset, self).__init__()
        self.input_transforms = transforms
        # self.images_info = []
        self.dataset_dir = dataset_dir
        self.images_dir = os.path.join(dataset_dir, 'images')
        self.images = []

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        query = Image.open(os.path.join(self.images_dir, self.images_info[index]['image_file']))
        if self.input_transforms:
            query = self.input_transforms(query)
        return query


# for image retrieval
class ImageDatabase(object):
    def __init__(self, images_info: list, images_dir: str, model, generate_database=False,
                 transforms=input_transforms_test(),
                 mode='retrieval'):
        self.model = model
        self.input_transforms = transforms
        self.database = None
        self.index = None
        self.mode = mode
        self.images_info = images_info.copy()
        self.images_dir = images_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        if generate_database:
            self._generate_database()

    @torch.no_grad()
    def _generate_database(self):
        assert len(self.images_info) > 0
        encodings = []
        # self.database = []
        print('Generating database from \'{}\'...'.format(self.images_dir))
        with torch.no_grad():
            for image_info in tqdm(self.images_info):
                image = Image.open(os.path.join(self.images_dir, image_info['image_file']))
                input = self.input_transforms(image).unsqueeze(0).to(self.device)
                netvlad_encoding = self.model(input).cpu().numpy().squeeze()
                image_info['encoding'] = netvlad_encoding
                # self.database.append({
                #     'image_file': image_filename,
                #     'encoding': netvlad_encoding,
                #     # 'position': xxx,
                #     # 'orientation': xxx,
                # })
                encodings.append(netvlad_encoding)
        dim_encoding = len(encodings[0])
        encodings = np.array(encodings)
        self.index = faiss.IndexFlatL2(dim_encoding)
        self.index.add(encodings)
        # self.model.cpu()
        self.images_info = np.array(self.images_info)
        print("Generation of database finished")

    # def query_image(self, netvlad_encoding, num_results=1):
    #     netvlad_encoding = netvlad_encoding.unsqueeze(0)
    #     distances, indices = self.index.search(netvlad_encoding, num_results)
    #     return self.database[indices[0]]

    def export_database(self, filename):
        np.save(filename, self.images_info)
        print('Exported database to {}'.format(filename))

    def import_database(self, filename):
        self.images_info = np.load(filename, allow_pickle=True)
        encodings = [datum['encoding'] for datum in self.images_info]
        dim_encoding = len(encodings[0])
        encodings = np.array(encodings)
        self.index = faiss.IndexFlatL2(dim_encoding)
        self.index.add(encodings)
        print('Imported database from {}'.format(filename))

    @torch.no_grad()
    def query_image(self, image_filename, num_results=1):
        assert len(self.images_info) > 0
        image = Image.open(image_filename)
        input = self.input_transforms(image).unsqueeze(0).to(self.device)
        netvlad_encoding = self.model(input).cpu().numpy()
        # netvlad_encoding = netvlad_encoding.unsqueeze(0)
        distances, indices = self.index.search(netvlad_encoding, num_results)
        return self.images_info[indices[0]]


if __name__ == '__main__':
    # dataset_dir = '/media/admini/My_data/0904/dataset'
    dataset_dir = '/home/li/Documents/wayz/image_data/dataset'
    dataset = NetVladDataset(dataset_dir, transforms=input_transforms())
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True)
    # data_loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn, shuffle=True)
    # with tqdm(data_loader) as tq:
    for item in data_loader:
        print(item)
    print(len(dataset))
