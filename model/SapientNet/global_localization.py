import sys
import os

sys.path.append(os.path.dirname(__file__))
print(sys.path)

import torch
# from SapientNet.Superglue import SuperGlue
from model.Superglue.superglue import SuperGlue

from sapientnet_with_dgcnn import DgcnnModel


def extract_descriptors():
    pass



def match_descriptors():


model = DgcnnModel(k=5, feature_dims=[64, 128, 256], emb_dims=[512, 256], output_classes=descriptor_dim)
model.load_state_dict(torch.load(os.path.join(DATA_DIR, "model-dgcnn-no-dropout.pth"), map_location=torch.device('cpu')))

super_glue_config = {
    'descriptor_dim': descriptor_dim,
    'weights': '',
    'keypoint_encoder': [32, 64, 128],
    'GNN_layers': ['self', 'cross'] * 6,
    'sinkhorn_iterations': 150,
    'match_threshold': 0.2,
}
superglue = SuperGlue(super_glue_config)
superglue.load_state_dict(torch.load(os.path.join(DATA_DIR, "Superglue-dgcnn-no-dropout.pth"), map_location=dev))

model.train()
superglue.train()
model = model.to(dev)
superglue = superglue.to(dev)

