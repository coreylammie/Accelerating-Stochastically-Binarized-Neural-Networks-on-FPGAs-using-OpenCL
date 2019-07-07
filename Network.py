import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from BinarizedModules import BinarizeLinear, BinarizeConv2d, STE


class BinarizedMNISTNetwork(nn.Module):

    def __init__(self):
        super(BinarizedMNISTNetwork, self).__init__()
        self.fc0 = BinarizeLinear(in_features=784, out_features=1000)
        self.bn0 = nn.BatchNorm1d(num_features=1000)
        self.act0 = nn.ReLU()
        self.fc1 = BinarizeLinear(in_features=1000, out_features=500)
        self.bn1 = nn.BatchNorm1d(num_features=500)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=500, out_features=10)

    def forward(self, input):
        x = input.view(input.size(0), -1)
        x = self.act0(self.bn0(self.fc0(x)))
        x = self.act1(self.bn1(self.fc1(x)))
        return F.log_softmax(self.fc2(x))

class BinarizedCIFARNetwork(nn.Module):

    def __init__(self, inflation_ratio=1):
        super(BinarizedCIFARNetwork, self).__init__()
        # conv2d(3, 128*inflation_ratio, kernel_size=5) -> batch_norm -> hard_tanh
        self.conv0 = BinarizeConv2d(in_channels=3, out_channels=128*inflation_ratio, kernel_size=3, stride=1, padding=1)
        self.bn0 = nn.BatchNorm2d(num_features=128*inflation_ratio)
        self.act0 = nn.ReLU()
        # conv2d(128*inflation_ratio, 128*inflation_ratio, kernel_size=3) -> mp(2) ->batch_norm ->  hard_tanh
        self.conv1 = BinarizeConv2d(in_channels=128*inflation_ratio, out_channels=128*inflation_ratio, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=128*inflation_ratio)
        self.act1 = nn.ReLU()
        # conv2d(128*inflation_ratio, 256*inflation_ratio, kernel_size=3) -> batch_norm -> hard_tanh
        self.conv2 = BinarizeConv2d(in_channels=128*inflation_ratio, out_channels=256*inflation_ratio, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=256*inflation_ratio)
        self.act2 = nn.ReLU()
        # conv2d(256*inflation_ratio, 256*inflation_ratio, kernel_size=3) -> mp(2) -> batch_norm -> hard_tanh
        self.conv3 = BinarizeConv2d(in_channels=256*inflation_ratio, out_channels=256*inflation_ratio, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=256*inflation_ratio)
        self.act3 = nn.ReLU()
        # conv2d(256*inflation_ratio, 512*inflation_ratio, kernel_size=3) -> batch_norm -> hard_tanh
        self.conv4 = BinarizeConv2d(in_channels=256*inflation_ratio, out_channels=512*inflation_ratio, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(num_features=512*inflation_ratio)
        self.act4 = nn.ReLU()
        # conv2d(512*inflation_ratio, 512*inflation_ratio, kernel_size=3) -> mp(2) -> batch_norm -> hard_tanh
        self.conv5 = BinarizeConv2d(in_channels=512*inflation_ratio, out_channels=512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(num_features=512)
        self.act5 = nn.ReLU()
        # linear(512 * 4 * 4) -> batch_norm -> hard_tanh
        self.fc6 = BinarizeLinear(in_features=512*4*4, out_features=1024)
        self.bn6 = nn.BatchNorm1d(num_features=1024)
        self.act6 = nn.ReLU()
        # linear(1024, 1024) -> batch_norm -> hard_tanh
        self.fc7 = BinarizeLinear(in_features=1024, out_features=1024)
        self.bn7 = nn.BatchNorm1d(num_features=1024)
        self.act7 = nn.ReLU()
        # linear(1024, 10)
        self.fc8 = nn.Linear(in_features=1024, out_features=10)

    def forward(self, input):
        x = self.act0(self.bn0(self.conv0(input)))
        x = self.act1(self.bn1(F.max_pool2d(self.conv1(x), 2)))
        x = self.act2(self.bn2(self.conv2(x)))
        x = self.act3(self.bn3(F.max_pool2d(self.conv3(x), 2)))
        x = self.act4(self.bn4(self.conv4(x)))
        x = self.act5(self.bn5(F.max_pool2d(self.conv5(x), 2)))
        x = x.view(x.size(0), -1)
        x = self.act6(self.bn6(self.fc6(x)))
        x = self.act7(self.bn7(self.fc7(x)))
        return F.log_softmax(self.fc8(x))
