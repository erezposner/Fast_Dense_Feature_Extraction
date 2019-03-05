import torch

from torch import nn


class BaseNet(nn.Module):
    def __init__(self):
        super(BaseNet, self).__init__()
        torch.manual_seed(10)
        torch.cuda.manual_seed(10)
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1)
        self.act1 = nn.ReLU()
        self.max_pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1)
        self.act2 = nn.ReLU()
        self.max_pool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(16, 128, kernel_size=2, stride=1)
        self.act3 = nn.ReLU()
        self.conv4 = nn.Conv2d(128, 113, kernel_size=1)

    def forward(self, x):
        torch.manual_seed(10)
        torch.cuda.manual_seed(10)
        x = self.conv1(x)
        x = self.act1(x)
        x = self.max_pool1(x)
        torch.manual_seed(10)
        torch.cuda.manual_seed(10)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.max_pool2(x)
        torch.manual_seed(10)
        torch.cuda.manual_seed(10)
        x = self.conv3(x)
        x = self.act3(x)
        torch.manual_seed(10)
        torch.cuda.manual_seed(10)
        y = self.conv4(x)
        return y
