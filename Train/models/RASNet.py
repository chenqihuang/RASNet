# -*- coding:utf-8 -*-
# !/ussr/bin/env python2
__author__ = "QiHuangChen"

import torch
import torch.nn as nn
import torch.nn.functional as F
from Train.train_config import *

class Residual_attention_Net(nn.Module):

    def __init__(self):
        super(Residual_attention_Net, self).__init__()
        self.res_attention = nn.Sequential(
            nn.Conv2d(256, 256, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, 3),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 1, 3),
        )
    def forward(self, x):
        return self.res_attention(x)


class Channel_attention_net(nn.Module):

    def __init__(self, channel=256, reduction=16):
        super(Channel_attention_net, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return y.expand_as(x)


class AlexNet_like(nn.Module):

    def __init__(self):
        super(AlexNet_like, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 96, 11, 2),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2))

        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, 5, 1, groups=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2))

        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 384, 3, 1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 384, 3, 1, groups=2),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True))

        self.conv5 = nn.Sequential(
            nn.Conv2d(384, 256, 3, 1, groups=2))

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        return conv5


class RASNet(nn.Module):

    def __init__(self):
        super(RASNet, self).__init__()
        self.feat_extraction = AlexNet_like()
        self.res_att = Residual_attention_Net()
        self.channel_att = Channel_attention_net()
        self.adjust = nn.Conv2d(1, 1, 1, 1)
        self.config = Config()
        self._initialize_weight()

    def forward(self, z, x):
        """
        forward pass
        z: examplare, BxCxHxW
        x: search region, BxCxH1xW1
        """
        # get features for z and x
        z_feat = self.feat_extraction(z)
        z_feat_channel = self.channel_att(z_feat)
        z_feat_residual = self.res_att(z_feat)
        # channel ,spatial
        z_feat = z_feat_channel * z_feat
        z_feat = z_feat_residual * z_feat

        z_feat = z_feat * z_feat_channel
        x_feat = self.feat_extraction(x)

        # correlation of z and z
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
            elif isinstance(m, nn.Linear):
                pass
        self.adjust.weight.data.fill_(1e-3)
        self.adjust.bias.data.zero_()

    def weight_loss(self, prediction, label, weight, penalty):
        """
        weighted cross entropy loss
        """
        loss = 0
        for i in range(label.size[0]):
            loss += F.binary_cross_entropy_with_logits(prediction[i],
                                               label[i],
                                               weight,
                                               size_average=False) * penalty
        return loss / self.config.batch_size

        # return F.binary_cross_entropy_with_logits(prediction,
        #                                           label,
        #                                           weight,
        #                                           size_average=False) / self.config.batch_size


if __name__ == "__main__":
    pass




