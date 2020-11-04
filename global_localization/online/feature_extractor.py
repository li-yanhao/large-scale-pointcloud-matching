from model.Superglue.superpoint import SuperPoint
import os
import torch
# import PIL.Image as Image
import torchvision.transforms as transforms

class FeatureExtractor(object):
    def __init__(self, config={}):
        super().__init__()
        default_config = {
            "images_dir": "/media/li/lavie/dataset/birdview_dataset/00",
            'superpoint': {
                'nms_radius': 4,
                'keypoint_threshold': 0.005,
                'max_keypoints': -1,
            },
            'saved_model_path': '/media/li/lavie/dataset/birdview_dataset/saved_models',
            # "resolution": 400,
        }

        config = {**default_config, **config}

        # self.resolution_ = config["resolution"]
        self.superpoint_  = SuperPoint(config['superpoint'])

        # saved_model_file_superpoint = os.path.join(config["saved_model_path"], 'superpoint-juxin.pth.tar')
        saved_model_file_superpoint = os.path.join(config["saved_model_path"], 'superpoint-rotation-invariant.pth.tar')

        model_checkpoint = torch.load(saved_model_file_superpoint, map_location=lambda storage, loc: storage)
        self.superpoint_.load_state_dict(model_checkpoint)

        self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.superpoint_.to(self.device_)
        self.superpoint_.eval()

    def extract_features(self, image):
        # Extract SuperPoint (keypoints, scores, descriptors) if not provided
        image_tensor = transforms.ToTensor()(image[...,None]).float()[None,...].to(self.device_)

        with torch.no_grad():
            pred = self.superpoint_({'image': image_tensor})
        return {k : [item.cpu() for item in v] for k, v in pred.items()}
