import os, sys
sys.path.append("../NetVLAD-pytorch")

import torch
import torch.nn as nn
from torch.autograd import Variable

from netvlad import NetVLAD, NetVLADPure
from netvlad import EmbedNet
from hard_triplet_loss import HardTripletLoss
from torchvision.models import resnet18


# Discard layers at the end of base network
# encoder = resnet18(pretrained=True)
# base_model = nn.Sequential(
#     encoder.conv1,
#     encoder.bn1,
#     encoder.relu,
#     encoder.maxpool,
#     encoder.layer1,
#     encoder.layer2,
#     encoder.layer3,
#     encoder.layer4,
# )
# dim = list(base_model.parameters())[-1].shape[0]  # last channels (512)
#
# # Define model for embedding
# net_vlad = NetVLAD(num_clusters=32, dim=dim, alpha=1.0)
# model = EmbedNet(base_model, net_vlad).cuda()
#
# # Define loss
# criterion = HardTripletLoss(margin=0.1).cuda()
#
# # This is just toy example. Typically, the number of samples in each classes are 4.
# labels = torch.randint(0, 10, (40, )).long()
# x = torch.rand(40, 3, 128, 128).cuda()
# output = model(x)
#
# triplet_loss = criterion(output, labels)

# input: N * D0

feature_dim = 256
batches = 40
N = 100
net_vlad = NetVLADPure(num_clusters=32, dim=feature_dim, alpha=1.0)

x = torch.rand(batches, feature_dim, N).cuda()
net_vlad = net_vlad.cuda()
y = net_vlad(x)
print(y)

# output: D1

