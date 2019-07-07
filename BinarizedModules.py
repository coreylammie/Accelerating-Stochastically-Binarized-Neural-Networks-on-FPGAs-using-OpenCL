import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd import Function
import numpy as np


def Binarize(x):
    return torch.sign(x)


class BinarizeLinear(nn.Linear):

    def __init__(self, *kargs, **kwargs):
        super(BinarizeLinear, self).__init__(*kargs, **kwargs)

    def forward(self, input):
        self.weight.transformed = Binarize(self.weight)
        out = nn.functional.linear(input, self.weight.transformed)
        if not self.bias is None:
            out += self.bias.view(1, -1).expand_as(out)

        return out


class BinarizeConv2d(nn.Conv2d):

    def __init__(self, scale_factor=1, *kargs, **kwargs):
        super(BinarizeConv2d, self).__init__(*kargs, **kwargs)

    def forward(self, input):
        self.weight.transformed = Binarize(self.weight)
        out = nn.functional.conv2d(input, self.weight.transformed, None, self.stride,
                                   self.padding, self.dilation, self.groups)

        if not self.bias is None:
            out += self.bias.view(1, -1, 1, 1).expand_as(out)

        return out


class STE(nn.Module):

    def __init__(self):
        super(STE, self).__init__()

    def forward(self, input):
        return torch.sign(input)
