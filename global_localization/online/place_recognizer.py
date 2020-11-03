import sys
sys.path.append("../../")
import numpy as np
import torch
import faiss
import torchvision.transforms as transforms
from model.Birdview.base_model import BaseModel
from model.Birdview.netvlad import NetVLAD
from model.Birdview.netvlad import EmbedNet
from tqdm import tqdm
from PIL import Image
import os


def input_transforms():
    return transforms.Compose([
        transforms.ToTensor(),
    ])

class PlaceRecognizer(object):
    def __init__(self, config={}, images_info=None, load_database=False):
        super().__init__()
        default_config = {
            'saved_model_path': '/media/li/lavie/dataset/birdview_dataset/saved_models',
            'num_clusters': 64,
            'final_dim': 256,
            'save_dir': None,
            'num_results': 3,

        }
        config = {**default_config, **config}

        base_model = BaseModel()
        net_vlad = NetVLAD(num_clusters=config["num_clusters"], dim=256, alpha=1.0, outdim=config["final_dim"])
        self.model_ = EmbedNet(base_model, net_vlad)
        saved_model_file = os.path.join(config["saved_model_path"], 'model-to-check-top1.pth.tar')
        model_checkpoint = torch.load(saved_model_file, map_location=lambda storage, loc: storage)
        self.model_.load_state_dict(model_checkpoint)

        self.save_dir_ = config['save_dir']
        self.images_info_ = [] if images_info is None else images_info
        self.index_ = faiss.IndexFlatL2(config['final_dim'])
        self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # self.device_ = torch.device("cpu")
        self.model_.to(self.device_)
        self.input_transforms_ = input_transforms()
        self.num_results_ = config["num_results"]

        if load_database:
            self._generate_database()
        pass
    
    # @torch.no_grad()
    # def save_spi(self, image):
    #     input = self.input_transforms_(image).unsqueeze(0).to(self.device_)
    #     netvlad_encoding = self.model_(input).cpu().numpy().squeeze()
    #     # image_info['encoding'] = netvlad_encoding
    #     # encodings.append(netvlad_encoding)
    #     spi_filename = "submap_" + str(len(self.images_info_)) + ".png"
    #     image_info = {
    #         'image_file': spi_filename,
    #         'encoding': netvlad_encoding,
    #         # 'position': pose[0:3, 3],
    #         # 'orientation': np.array([1,0,0,0])
    #     }
    #     self.images_info_.append(image_info)
    #     self.index_.add(netvlad_encoding[None, ...])
    #
    #     print("Saved SPI ", spi_filename)
    #     pass
    
    @torch.no_grad()
    def query_spi(self, image):
        input = self.input_transforms_(image).unsqueeze(0).to(self.device_)
        with torch.no_grad():
            netvlad_encoding = self.model_(input).cpu().numpy() # 1, D
        result_images_info = None

        # search spi in database
        if len(self.images_info_) >= self.num_results_:
            distances, indices = self.index_.search(netvlad_encoding, self.num_results_)
            result_images_info = [self.images_info_[index] for index in indices[0]]

        # save the new encoding
        spi_filename = "submap_" + str(len(self.images_info_)) + ".png"
        image_info = {
            'image_file': spi_filename,
            'encoding': netvlad_encoding,
            # 'position': pose[0:3, 3],
            # 'orientation': np.array([1, 0, 0, 0])
        }
        self.images_info_.append(image_info)
        self.index_.add(netvlad_encoding)
        # print("Saved SPI ", netvlad_encoding.shape)

        return result_images_info

    @torch.no_grad()
    def extract_descriptor(self, image):
        input = self.input_transforms_(image).unsqueeze(0).to(self.device_)
        with torch.no_grad():
            netvlad_encoding = self.model_(input).cpu().numpy() # 1, D
        return netvlad_encoding

    @torch.no_grad()
    def _generate_database(self):
        assert len(self.images_info_) > 0
        encodings = []
        # self.database = []
        print('Generating database from \'{}\'...'.format(self.images_dir))
        with torch.no_grad():
            for image_info in tqdm(self.images_info):
                image = Image.open(os.path.join(self.images_dir, image_info['image_file']))
                input = self.input_transforms(image).unsqueeze(0).to(self.device)
                netvlad_encoding = self.model_(input).cpu().numpy().squeeze()
                image_info['encoding'] = netvlad_encoding
                encodings.append(netvlad_encoding)

        dim_encoding = len(encodings[0])
        encodings = np.array(encodings)
        self.index = faiss.IndexFlatL2(dim_encoding)
        self.index.add(encodings)
        self.images_info = np.array(self.images_info)
        print("Generation of database finished")

    def export_database(self, filename):
        np.save(filename, self.images_info_)
        print('Exported database to {}'.format(filename))
    
    # def import_database(self, filename):
    #     self.images_info_ = np.load(filename, allow_pickle=True)
    #     encodings = [datum['encoding'] for datum in self.images_info_]
    #     dim_encoding = len(encodings[0])
    #     encodings = np.array(encodings)
    #     self.index_.add(encodings)
    #     print('Imported database from {}'.format(filename))

