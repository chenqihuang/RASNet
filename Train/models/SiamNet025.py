# -*- coding:utf-8 -*-
# !/ussr/bin/env python2
__author__ = "QiHuangChen"

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F


class AlexNet_like(nn.Module):
    def __init__(self):
        super(AlexNet_like, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 24, 11, 2),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2))

        self.conv2 = nn.Sequential(
            nn.Conv2d(24, 64, 5, 1, groups=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2))

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 96, 3, 1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(
            nn.Conv2d(96, 96, 3, 1, groups=2),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True))

        self.conv5 = nn.Sequential(
            nn.Conv2d(96, 64, 3, 1, groups=2))

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        return conv5


class SiamNet(nn.Module):

    def __init__(self):
        super(SiamNet, self).__init__()
        self.feat_extraction = AlexNet_like()
        self.adjust = nn.Sequential(nn.Conv2d(1, 1, 1, 1))
        self._initialize_weight()

    def forward(self, z, x):
        """
        forward pass
        z: examplare, BxCxHxW
        x: search region, BxCxH1xW1
        """
        # get features for z and x
        z_feat = self.feat_extraction(z)
        x_feat = self.feat_extraction(x)
        xcorr_out = self.xcorr(z_feat, x_feat)
        score = self.adjust(xcorr_out)

        return score

    def xcorr(self, z, x):
        """
        correlation layer as in the original SiamFC (convolution process in fact)
        """
        batch_size_x, channel_x, w_x, h_x = x.shape
        x = torch.reshape(x, (1, batch_size_x * channel_x, w_x, h_x))

        # group convolution
        out = F.conv2d(x, z, groups = batch_size_x)

        batch_size_out, channel_out, w_out, h_out = out.shape
        xcorr_out = torch.reshape(out, (channel_out, batch_size_out, w_out, h_out))

        return xcorr_out

    def _initialize_weight(self):
        """
        initialize network parameters
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        self.adjust.weight.data.fill_(1e-3)
        self.adjust.bias.data.zero_()


if __name__ == "__main__":
    pass




