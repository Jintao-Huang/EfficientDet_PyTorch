# Author: Jintao Huang
# Time: 2020-5-19

import torch.nn as nn
import torch
from .bifpn import DepthSeparableConv2d
from .efficientnet import Swish
from collections import OrderedDict


class Classifier(nn.Module):
    def __init__(self, in_channels, num_anchors, num_classes, num_repeat, bn_momentum, bn_eps, norm_layer=None):
        """类似于Regressor

        out_channels = num_anchors * num_classes"""
        super(Classifier, self).__init__()
        norm_layer = norm_layer or nn.BatchNorm2d
        self.num_repeat = num_repeat
        self.num_anchors = num_anchors
        self.num_classes = num_classes

        conv_list = []
        bn_list = []
        for _ in range(num_repeat):
            conv_list.append(DepthSeparableConv2d(in_channels, in_channels, 3, 1, False))

        for _ in range(5):  # 5个features
            for _ in range(num_repeat):
                bn_list.append(norm_layer(in_channels, bn_eps, bn_momentum))

        self.conv_list = nn.ModuleList(conv_list)
        self.bn_list = nn.ModuleList(bn_list)  # 5 * num_repeat
        out_channels = num_anchors * num_classes
        self.head = DepthSeparableConv2d(in_channels, out_channels, 3, 1, True)
        self.swish = Swish()

    def forward(self, features):
        """

        :param features: List(P3_n, P4_n, P5_n, P6_n, P7_n)
        :return: Tensor[N, NUM, 4]. NUM: F*H*W*A
        """
        features = features.copy()
        output = []
        if isinstance(features, OrderedDict):
            features = list(features.values())
        assert len(features) == 5, "len(features) != 5"
        for i in range(5):
            x = features.pop(0)
            for j in range(self.num_repeat):
                conv = self.conv_list[j]
                bn = self.bn_list[i * self.num_repeat + j]
                x = self.swish(bn(conv(x)))

            x = self.head(x)
            x = x.permute((0, 2, 3, 1))  # N, H, W, C
            x = torch.reshape(x, (x.shape[0], -1, self.num_classes))
            output.append(x)
        return torch.sigmoid(torch.cat(output, dim=1))


class Regressor(nn.Module):
    def __init__(self, in_channels, num_anchors, num_repeat, bn_momentum, bn_eps, norm_layer=None):
        """out_channels = num_anchors * 4"""
        super(Regressor, self).__init__()

        norm_layer = norm_layer or nn.BatchNorm2d
        self.num_repeat = num_repeat

        conv_list = []
        bn_list = []
        for _ in range(num_repeat):
            conv_list.append(DepthSeparableConv2d(in_channels, in_channels, 3, 1, False))

        for _ in range(5):  # 5个features
            for _ in range(num_repeat):
                bn_list.append(norm_layer(in_channels, bn_eps, bn_momentum))

        self.conv_list = nn.ModuleList(conv_list)
        self.bn_list = nn.ModuleList(bn_list)  # 5 * num_repeat
        out_channels = num_anchors * 4
        self.head = DepthSeparableConv2d(in_channels, out_channels, 3, 1, True)
        self.swish = Swish()

    def forward(self, features):
        """

        :param features: List(P3_n, P4_n, P5_n, P6_n, P7_n)
        :return: Tensor[N, NUM, 4]. NUM: F*H*W*A
        """
        features = features.copy()
        output = []
        if isinstance(features, OrderedDict):
            features = list(features.values())
        assert len(features) == 5, "len(features) != 5"
        for i in range(5):
            x = features.pop(0)
            for j in range(self.num_repeat):
                conv = self.conv_list[j]
                bn = self.bn_list[i * self.num_repeat + j]
                x = self.swish(bn(conv(x)))

            x = self.head(x)
            x = x.permute((0, 2, 3, 1))  # N, H, W, C
            x = torch.reshape(x, (x.shape[0], -1, 4))
            output.append(x)
        return torch.cat(output, dim=1)
