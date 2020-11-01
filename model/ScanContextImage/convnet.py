import torch
import torch.nn as nn


class ConvNet(nn.Module):
    def __init__(self, width=1000, height=1000):
        super(ConvNet, self).__init__()
        self.width = width
        self.height= height

        self.Drop1 = 0.7
        self.Drop2 = 0.7

        self.KernelSize = 5

        self.nConv1Filter = 64
        self.nConv2Filter = 128
        self.nConv3Filter = 256

        self.nFCN1 = 64

        # self.dim_1 = 64
        # self.dim_2 = 128
        # self.dim_3 = 256
        # self.dim_4 = 64
        # self.dim_out = 64
        # self.max_value = 256.0
        #
        # kernel_size = 15

        # self.layer_3_out_width = (((width - kernel_size+1) // 2 - kernel_size+1) // 2 - kernel_size+1) // 2
        # self.layer_3_out_height = (((height - kernel_size+1) // 2 - kernel_size+1) // 2 - kernel_size+1) // 2

        self.layer_1 = nn.Sequential(
            nn.Conv2d(3, self.nConv1Filter, (self.KernelSize, self.KernelSize)),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=0),
            # nn.BatchNorm2d(self.dim_1, affine=False),
            nn.BatchNorm2d(self.nConv1Filter),
        )

        self.layer_2 = nn.Sequential(
            nn.Conv2d(self.nConv1Filter, self.nConv2Filter, (self.KernelSize, self.KernelSize)),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            # nn.BatchNorm2d(self.dim_2, affine=False),
            nn.BatchNorm2d(self.nConv2Filter),
        )

        self.layer_3 = nn.Sequential(
            nn.Conv2d(self.nConv2Filter, self.nConv3Filter, (self.KernelSize, self.KernelSize)),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Flatten()
        )

        self.fc_1 = nn.Sequential(
            nn.Dropout(p=0.7),
            nn.Linear(self.dim_3 * self.layer_3_out_width * self.layer_3_out_height, self.nFCN1),
        )

        self.fc_2 = nn.Sequential(
            nn.Dropout(p=0.7),
            nn.Linear(self.dim_4, self.dim_out),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        h = x / self.max_value
        h = self.layer_1(h)
        h = self.layer_2(h)
        h = self.layer_3(h)
        h = self.fc_1(h)
        h = self.fc_2(h)
        return h


if __name__ == "__main__":
    W, H = 400, 400
    base_model = ConvNet(W, H)
    input = torch.randn(3, 1, W, H)
    output = base_model(input)
    print(output.shape)