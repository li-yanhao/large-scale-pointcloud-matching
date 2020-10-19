import torch
import torch.nn as nn


class BaseModel(nn.Module):
    def __init__(self, width=1000, height=1000):
        super(BaseModel, self).__init__()
        self.width = width
        self.height= height
        self.dim_1 = 64
        self.dim_2 = 128
        self.dim_3 = 256
        self.dim_4 = 64
        self.dim_out = 64
        self.max_value = 256.0

        kernel_size = 15

        self.layer_3_out_width = (((width - kernel_size+1) // 2 - kernel_size+1) // 2 - kernel_size+1) // 2
        self.layer_3_out_height = (((height - kernel_size+1) // 2 - kernel_size+1) // 2 - kernel_size+1) // 2

        self.layer_1 = nn.Sequential(
            nn.Conv2d(1, self.dim_1, (kernel_size,kernel_size), 1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.BatchNorm2d(self.dim_1, affine=False),
        )

        self.layer_2 = nn.Sequential(
            nn.Conv2d(self.dim_1, self.dim_2, (kernel_size, kernel_size), 1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.BatchNorm2d(self.dim_2, affine=False),
        )

        self.layer_3 = nn.Sequential(
            nn.Conv2d(self.dim_2, self.dim_3, (kernel_size, kernel_size), 1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            # nn.Flatten()
        )

        self.fc_1 = nn.Sequential(
            nn.Dropout(p=0.7),
            nn.Linear(self.dim_3 * self.layer_3_out_width * self.layer_3_out_height, self.dim_4),
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
        # h = self.fc_1(h)
        # h = self.fc_2(h)
        return h


if __name__ == "__main__":
    W, H = 1000, 1000
    base_model = BaseModel(W, H)
    input = torch.randn(3, 1, W, H)
    output = base_model(input)
    print(output.shape)