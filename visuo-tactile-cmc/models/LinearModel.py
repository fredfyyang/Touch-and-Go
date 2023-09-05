from __future__ import print_function
from torch import channel_shuffle
import torchvision

import torch.nn as nn
import torch


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, feat):
        return feat.view(feat.size(0), -1)

class LinearClassifierResNet(nn.Module):
    def __init__(self, layer=6, n_label=1000):
        super(LinearClassifierResNet, self).__init__()
        self.layer = layer
        if layer == 1:
            nChannels = 64
        elif layer == 2:
            nChannels = 64
        elif layer == 3:
            nChannels = 128
        elif layer == 4:
            nChannels = 256
        elif layer == 5:
            nChannels = 512
        elif layer == 6:
            nChannels = 512
        else:
            raise NotImplementedError('layer not supported: {}'.format(layer))

        self.classifier = nn.Sequential()
        self.classifier.add_module('LiniearClassifier', nn.Linear(nChannels, n_label))
        self.initilize()

    def initilize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.fill_(0.0)

    def forward(self, x):
        if self.layer < 6:
            avg_pool = nn.AvgPool2d((x.shape[2], x.shape[3]))
            x = avg_pool(x).squeeze()
        return self.classifier(x)