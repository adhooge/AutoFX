"""
ResNet block based on https://pytorch-tutorial.readthedocs.io/en/latest/tutorial/chapter03_intermediate/3_2_2_cnn_resnet_cifar10/
Reference:
https://arxiv.org/pdf/1512.03385.pdf
"""

import torch
from torch import nn


class Downsampler(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(Downsampler, self).__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, padding=1),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        out = self.downsample(x)
        return out


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride: int = 1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size//2)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        if self.downsample:
            residual = self.downsample(x)
        out = out + residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, num_params):
        super(ResNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.block1_1 = ResNetBlock(64, 64, 3)
        self.block1_2 = ResNetBlock(64, 64, 3)
        self.block2_1 = ResNetBlock(64, 128, 3, 2, Downsampler(64, 128, 2))
        self.block2_2 = ResNetBlock(128, 128, 3)
        self.block3_1 = ResNetBlock(128, 256, 3, 2, Downsampler(128, 256, 2))
        self.block3_2 = ResNetBlock(256, 256, 3)
        self.avgpool = nn.AvgPool2d(8)
        self.fcl = nn.Linear(22144, num_params)

    def forward(self, x):
        out = self.conv1(x)
        out = self.maxpool(out)
        out = self.block1_1(out)
        out = self.block1_2(out)
        out = self.block2_1(out)
        out = self.block2_2(out)
        out = self.block3_1(out)
        out = self.block3_2(out)
        out = self.avgpool(out)
        out = torch.flatten(out, start_dim=1)
        out = self.fcl(out)
        return out
