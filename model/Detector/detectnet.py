import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageDraw
import numpy as np
from scipy import ndimage
import torch.optim as optim


class DetectNet(nn.Module):
    def __init__(self, width=1000, height=1000):
        super(DetectNet, self).__init__()
        self.width = width
        self.height= height
        self.dim_1 = 64
        self.dim_2 = 256
        self.dim_3 = 65

        self.max_value = 256.0

        kernel_size = 10

        self.layer_3_out_width = (((width) // 2) // 2 ) // 2
        self.layer_3_out_height = (((height) // 2 ) // 2 ) // 2

        self.layer_1 = nn.Sequential(
            nn.Conv2d(1, self.dim_1, (kernel_size,kernel_size), 1, padding=kernel_size//2),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.BatchNorm2d(self.dim_1, affine=False),
        )

        self.layer_2 = nn.Sequential(
            nn.Conv2d(self.dim_1, self.dim_2, (kernel_size, kernel_size), 1, padding=kernel_size//2),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.BatchNorm2d(self.dim_2, affine=False),
        )

        self.layer_3 = nn.Sequential(
            nn.Conv2d(self.dim_2, self.dim_3, (kernel_size, kernel_size), 1, padding=kernel_size//2),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Softmax(dim=1)
            # nn.Flatten()
        )

        # self.fc_1 = nn.Sequential(
        #     nn.Dropout(p=0.7),
        #     nn.Linear(self.dim_3 * self.layer_3_out_width * self.layer_3_out_height, self.dim_4),
        # )
        #
        # self.fc_2 = nn.Sequential(
        #     nn.Dropout(p=0.7),
        #     nn.Linear(self.dim_4, self.dim_out),
        #     nn.Softmax(dim=1)
        # )

    def forward(self, x):
        h = x / self.max_value
        h = self.layer_1(h)
        h = self.layer_2(h)
        h = self.layer_3(h)

        unwarped_img = h[:, :64, :, :]
        dustbins = h[:, -1:, :, :]

        warped_img = unwarped_img.view(B, 8, 8, W // 8, H // 8)
        warped_img = warped_img.permute(0, 3, 1, 4, 2)
        warped_img = warped_img.reshape(B, W, H)
        dustbins = dustbins.view(B, W // 8, H // 8)
        # h = self.fc_1(h)
        # h = self.fc_2(h)
        return warped_img, dustbins


def random_transform(x, degree, translation):
    # input: torch.Tensor, B * C * W * H
    # output: torch.Tensor, B * C * W * H
    x = np.array(x)
    num_dim = len(x.shape)
    x = ndimage.rotate(x, degree, axes=(num_dim-2, num_dim-1), reshape=False)
    x = ndimage.shift(x, [0, 0, translation[0], translation[1]], cval=0)
    return torch.Tensor(x)


if __name__ == "__main__":
    dev = torch.device('cuda')
    B, W, H = 1, 200, 200
    detect_net = DetectNet(W, H)
    input = torch.randn(B, 1, W, H)

    input_raw = Image.open('/media/admini/lavie/dataset/birdview_dataset/00/submap_12.png')
    input_raw.show()
    tf = transforms.Compose([
        transforms.Resize((W, H)),
        transforms.ToTensor(),
    ])
    input_raw = tf(input_raw).unsqueeze(0)

    detect_net.to(dev)
    optimizer = optim.Adam(detect_net.parameters(), lr=0.001)

    epochs = 100
    K = 256
    for epoch in range(epochs):
        optimizer.zero_grad()
        degree = np.random.randn(1)[0] * 360
        input_p = random_transform(input_raw, degree, [0,0]).to(dev)
        input = input_raw.to(dev)

        warped_img, dustbins = detect_net(input)
        # print(warped_img.shape)
        # print(dustbins.shape)


        warped_img_p, dustbins_p = detect_net(input_p)

        # print(warped_img_p.shape)
        # print(dustbins_p.shape)

        # tf = transforms.Compose([
        #     transforms.ToPILImage(),
        #     transforms.RandomAffine(50, translate=(0.1, 0.2), scale=None, shear=None, resample=Image.BILINEAR),
        #     transforms.ToTensor(),
        # ])
        # tf = transforms.RandomAffine(50, translate=(0.1, 0.2), scale=None, shear=None, resample=Image.BILINEAR)
        # tf(input)

        # loss function

        scores = warped_img.view(B, -1)
        scores_p = warped_img_p.view(B, -1)
        top_scores = torch.topk(scores, K)
        top_scores_p = torch.topk(scores_p, K)

        POI_indices = np.array(top_scores[1].cpu())
        POI_bitmap = np.zeros((W * H), dtype=bool)
        POI_bitmap[POI_indices.reshape(-1)] = True
        POI_bitmap = POI_bitmap.reshape(B, 1, W, H)
        POI_bitmap = random_transform(POI_bitmap, degree, [0, 0])
        POI_bitmap = POI_bitmap.reshape(-1)
        true_indices = np.where(POI_bitmap > 0)

        top_scores_p[1].cpu()

        matches_indices = np.intersect1d(top_scores[1].cpu(), top_scores_p[1].cpu())


        loss = scores_p[0][matches_indices]
        # loss = - top_scores[0][matches_indices].sum() - top_scores_p[0][matches_indices].sum() \
        #        + top_scores[0][~matches_indices].sum() + top_scores_p[0][~matches_indices].sum()
        loss.backward()
        optimizer.step()

        print('loss: {}'.format(loss.item()))

    # visualize interest points
    input = input_raw.to(dev)
    warped_img, _ = detect_net(input)
    scores = warped_img.view(B, -1)
    top_scores = torch.topk(scores, K)

    POI_indices = np.array(top_scores[1].cpu())
    POI_bitmap = np.zeros((W*H), dtype=bool)
    POI_bitmap[POI_indices.reshape(-1)] = 255
    POI_bitmap = POI_bitmap.reshape(W, H)
    img = Image.fromarray(POI_bitmap)
    img.show()
