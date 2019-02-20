import torch

from torch import nn

class BaseNet(nn.Module):
    def __init__(self):
        super(BaseNet, self).__init__()
        torch.manual_seed(10)
        torch.cuda.manual_seed(10)
        self.conv_1 = nn.Conv2d(1, 8, kernel_size=3,stride=1)
        self.act_1 = nn.ReLU()
        self.max_pool_1 = nn.MaxPool2d(kernel_size=2)
        self.conv_2 = nn.Conv2d(8, 16, kernel_size=3,stride=1)
        self.act_2 = nn.ReLU()
        self.max_pool_2 = nn.MaxPool2d(kernel_size=2)
        self.conv_3 = nn.Conv2d(16, 128, kernel_size=2,stride=1)
        self.act_3 = nn.ReLU()
        self.conv_4 = nn.Conv2d(128, 113, kernel_size=1)

    def forward(self, x):
        torch.manual_seed(10)
        torch.cuda.manual_seed(10)
        x = self.conv_1(x)
        x = self.act_1(x)
        x = self.max_pool_1(x)
        torch.manual_seed(10)
        torch.cuda.manual_seed(10)
        x = self.conv_2(x)
        x = self.act_2(x)
        x = self.max_pool_2(x)
        torch.manual_seed(10)
        torch.cuda.manual_seed(10)
        x = self.conv_3(x)
        x = self.act_3(x)
        torch.manual_seed(10)
        torch.cuda.manual_seed(10)
        y = self.conv_4(x)
        return y

