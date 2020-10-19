from model.SapientNet.superpoint import *
from PIL import Image
from torchvision.transforms import transforms

def superpoint_test():
    super_config = {
        'descriptor_dim': 256,
        'nms_radius': 4,
        'keypoint_threshold': 0.005,
        'max_keypoints': -1,
        'remove_borders': 4,
    }

    superpoint = SuperPoint(super_config)

    image = Image.open('/media/admini/lavie/dataset/birdview_dataset/00/submap_1445.png')
    tf = transforms.Compose([
        transforms.Resize(size=(300, 300)),
        # transforms.RandomResizedCrop(size=(600, 960), scale=(0.8, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=2),
        # transforms.RandomRotation(degrees=360),
        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor(),
    ])
    image = tf(image)
    image = image.unsqueeze(0)
    data = {
        'image': image
    }

    out = superpoint(data)


if __name__ == "__main__":
    superpoint_test()