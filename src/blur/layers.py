# coding: utf-8

import torch.nn as nn
from torch.autograd import Variable
import torch


class SpatialScale2d(nn.Module):
    def __init__(self, shape):
        """
        :param shape: (width,height)
        """
        super(SpatialScale2d, self).__init__()
        assert len(shape) == 2
        self.weight = nn.Parameter(torch.zeros(*shape))

    def forward(self, x):
        return x * self.weight


class ConstantScale(nn.Module):
    def __init__(self):
        super(ConstantScale, self).__init__()
        self.scale = nn.Parameter(torch.ones(1.0))

    def forward(self, x):
        return x * self.scale


if __name__ == '__main__':
    pass
