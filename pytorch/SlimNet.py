import torch

from torch import nn
from pytorch.FDFE import multiPoolPrepare, multiMaxPooling, unwrapPrepare, unwrapPool


class SlimNet(nn.Module):
    def __init__(self, base_net, pH, pW, sL1, sL2, imH, imW):
        super(SlimNet, self).__init__()
        self.imH = imH
        self.imW = imW
        self.multiPoolPrepare = multiPoolPrepare(pH, pW)
        self.conv1 = list(base_net.modules())[1]
        self.act1 = list(base_net.modules())[2]
        self.multiMaxPooling1 = multiMaxPooling(sL1, sL1, sL1, sL1)
        self.conv2 = list(base_net.modules())[4]
        self.act2 = list(base_net.modules())[5]
        self.multiMaxPooling2 = multiMaxPooling(sL2, sL2, sL2, sL2)
        self.conv3 = list(base_net.modules())[7]
        self.act3 = list(base_net.modules())[8]
        self.conv4 = list(base_net.modules())[9]
        self.outChans = list(base_net.modules())[9].out_channels
        self.unwrapPrepare = unwrapPrepare()
        self.unwrapPool2 = unwrapPool(self.outChans, imH / (sL1 * sL2), imW / (sL1 * sL2), sL2, sL2)
        self.unwrapPool3 = unwrapPool(self.outChans, imH / sL1, imW / sL1, sL1, sL1)

    def forward(self, x):
        import numpy as np
        x = self.multiPoolPrepare(x)
        torch.manual_seed(10)
        torch.cuda.manual_seed(10)
        x = self.conv1(x)
        x = self.act1(x)
        x = self.multiMaxPooling1(x)

        torch.manual_seed(10)
        torch.cuda.manual_seed(10)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.multiMaxPooling2(x)
        torch.manual_seed(10)
        torch.cuda.manual_seed(10)
        x = self.conv3(x)
        x = self.act3(x)
        x = self.conv4(x)
        x = self.unwrapPrepare(x)
        x = self.unwrapPool2(x)
        x = self.unwrapPool3(x)
        y = x.view(-1, self.imH, self.imW)

        return y

