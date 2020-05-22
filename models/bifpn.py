# Author: Jintao Huang
# Time: 2020-5-18

import torch
import torch.nn as nn
from .efficientnet import get_same_padding, Swish
import torch.nn.functional as F
from collections import OrderedDict


class Conv2dSamePadding(nn.Conv2d):
    """Conv2dDynamicSamePadding

    由于输入大小都是128的倍数，所以动态卷积和静态卷积的结果是一致的。此处用动态卷积代替静态卷积，因为实现方便。

    Since the input size is a multiple of 128,
    the results of dynamic convolution and static convolution are consistent.
    Here, dynamic convolution is used instead of static convolution,
    because it is convenient to implement"""

    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, bias):
        self.kernel_size = kernel_size
        self.stride = stride
        super(Conv2dSamePadding, self).__init__(
            in_channels, out_channels, kernel_size, stride, groups=groups, bias=bias
        )

    def forward(self, x):
        padding = get_same_padding(x.shape[-2:], self.kernel_size, self.stride)
        x = F.pad(x, padding)
        x = super().forward(x)
        return x


class MaxPool2dSamePadding(nn.MaxPool2d):
    """MaxPool2dDynamicSamePadding

    由于输入大小都是128的倍数，所以动态池化和静态池化的结果是一致的。此处用动态池化代替静态池化，因为实现方便。

    Since the input size is a multiple of 128,
    the results of dynamic maxpool and static maxpool are consistent.
    Here, dynamic maxpool is used instead of static maxpool,
    because it is convenient to implement"""

    def __init__(self, kernel_size, stride):
        self.kernel_size = kernel_size
        self.stride = stride
        super(MaxPool2dSamePadding, self).__init__(
            kernel_size, stride
        )

    def forward(self, x):
        padding = get_same_padding(x.shape[-2:], self.kernel_size, self.stride)
        x = F.pad(x, padding)
        x = super().forward(x)
        return x


class DepthSeparableConv2d(nn.Sequential):
    """depthwise separable convolution"""

    def __init__(self, in_channels, out_channels, kernel_size, stride):
        depthwise_conv = Conv2dSamePadding(in_channels, in_channels, kernel_size, stride, in_channels, False)
        pointwise_conv = Conv2dSamePadding(in_channels, out_channels, 1, 1, 1, True)  # 可改为False
        super(DepthSeparableConv2d, self).__init__(
            OrderedDict({
                "depthwise_conv": depthwise_conv,
                "pointwise_conv": pointwise_conv
            })
        )


class BiFPNBlock(nn.Module):
    def __init__(self, in_channels_list, fpn_channels, attention, attention_eps, bn_momentum, bn_eps, norm_layer=None):
        super(BiFPNBlock, self).__init__()
        self.attention = attention
        self.attention_eps = attention_eps
        norm_layer = norm_layer or nn.BatchNorm2d
        # create modules
        if isinstance(in_channels_list, (tuple, list)):  # first BiFPN block
            # generate P6 and P7
            self.in_blocks = nn.ModuleDict(OrderedDict({
                "to_P3_0": nn.Sequential(  # P3
                    Conv2dSamePadding(in_channels_list[0], fpn_channels, 1, 1, 1, True),
                    norm_layer(fpn_channels, bn_eps, bn_momentum),
                ),
                "to_P4_0": nn.Sequential(  # P4
                    Conv2dSamePadding(in_channels_list[1], fpn_channels, 1, 1, 1, True),
                    norm_layer(fpn_channels, bn_eps, bn_momentum),
                ),
                "to_P5_0": nn.Sequential(  # P5
                    Conv2dSamePadding(in_channels_list[2], fpn_channels, 1, 1, 1, True),
                    norm_layer(fpn_channels, bn_eps, bn_momentum),
                ),
                "to_P6_0": nn.Sequential(
                    Conv2dSamePadding(in_channels_list[2], fpn_channels, 1, 1, 1, True),
                    norm_layer(fpn_channels, bn_eps, bn_momentum),
                    MaxPool2dSamePadding(3, 2)
                ),
                "to_P7_0": MaxPool2dSamePadding(3, 2),
                # P4, P5的第二条出线 (直连). P4 and P5 has two outputs
                "to_P4_02": nn.Sequential(  # P4
                    Conv2dSamePadding(in_channels_list[1], fpn_channels, 1, 1, 1, True),
                    norm_layer(fpn_channels, bn_eps, bn_momentum),
                ),
                "to_P5_02": nn.Sequential(  # P5
                    Conv2dSamePadding(in_channels_list[2], fpn_channels, 1, 1, 1, True),
                    norm_layer(fpn_channels, bn_eps, bn_momentum),
                ),
                # P6_2 使用 P6的输出(P6_2 uses the output of P6)
            }))
        conv_block1 = []  # "P6_0_to_P6_1", "P5_0_to_P5_1", "P4_0_to_P4_1", "P3_0_to_P3_2"
        conv_block2 = []  # "P4_1_to_P4_2", "P5_1_to_P5_2", "P6_1_to_P6_2", "P7_0_to_P7_2"
        upsample_block = []  # "P7_0_to_P6_1", "P6_1_to_P5_1", "P5_1_to_P4_1", "P4_1_to_P3_2"
        downsample_block = []  # "P3_2_to_P4_2", "P4_2_to_P5_2", "P5_2_to_P6_2", "P6_2_to_P7_2"
        for _ in range(4):
            conv_block1.append(nn.Sequential(
                DepthSeparableConv2d(fpn_channels, fpn_channels, 3, 1),
                norm_layer(fpn_channels, bn_eps, bn_momentum),
            ))
            conv_block2.append(nn.Sequential(
                DepthSeparableConv2d(fpn_channels, fpn_channels, 3, 1),
                norm_layer(fpn_channels, bn_eps, bn_momentum),
            ), )
            upsample_block.append(nn.UpsamplingNearest2d(scale_factor=2))
            downsample_block.append(MaxPool2dSamePadding(3, 2))

        self.conv_block1 = nn.ModuleList(conv_block1)
        self.conv_block2 = nn.ModuleList(conv_block2)
        self.upsample_block = nn.ModuleList(upsample_block)
        self.downsample_block = nn.ModuleList(downsample_block)
        self.swish = Swish()

        # extra weight
        if attention:
            self.weight_relu = nn.ReLU()

            self.to_P6_1_w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
            self.to_P5_1_w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
            self.to_P4_1_w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
            self.to_P3_1_w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)

            self.to_P4_2_w = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
            self.to_P5_2_w = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
            self.to_P6_2_w = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
            self.to_P7_2_w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)

    def forward(self, x):
        """

        :param x: List/OrderDict(P3, P4, P5)
        :return: List(P3_2, P4_2, P5_2, P6_2, P7_2)
        """
        if isinstance(x, OrderedDict):
            x = list(x.values())
        if hasattr(self, 'in_blocks'):
            P3, P4, P5 = x
            # Generate P6_0, P7_0
            P6_0 = self.in_blocks["to_P6_0"](P5)
            P7_0 = self.in_blocks["to_P7_0"](P6_0)
            # Adjust P3, P4, and P5 dimensions -> P3_0, P4_0, P5_0
            in_blocks = list(self.in_blocks.values())[:3]
            for i in range(3):
                x.append(in_blocks[i](x.pop(0)))
            # x: [P3_0, P4_0, P5_0]
            # ---------------------
            x += [P6_0, P7_0]
            del P3, P6_0, P7_0
        # x: [P3_0, P4_0, P5_0, P6_0, P7_0]
        # --------------------------------
        # calculate P6_1, P5_1, P4_1, P3_2
        out_1 = []
        conv_block1 = self.conv_block1
        upsample_block = self.upsample_block
        in_1 = list(reversed(x))  # [P6_0, P5_0, P4_0, P3_0]
        in_2 = [x[-1]]  # [P7_0]
        if self.attention:
            weights_1 = [self.to_P6_1_w, self.to_P5_1_w, self.to_P4_1_w, self.to_P3_1_w]
            for i in range(4):
                weight = self.weight_relu(weights_1[i])
                weight = weight / (torch.sum(weight, dim=0) + self.attention_eps)  # normalize
                out_1.append(conv_block1[i](self.swish(weight[0] * in_1[i + 1] +
                                                       weight[1] * upsample_block[i](in_2[i]))))
                in_2.append(out_1[-1])
            del weights_1
        else:
            for i in range(4):
                out_1.append(conv_block1[i](self.swish(in_1[i + 1] + upsample_block[i](in_2[i]))))
                in_2.append(out_1[-1])
        del in_1, in_2  # Prevent interference with subsequent parameter references

        # out_1: [P6_1, P5_1, P4_1, P3_2]
        # --------------------------------
        # x: [P3_0, P4_0, P5_0, P6_0, P7_0]
        # calculate P4_02, P5_02
        if hasattr(self, 'in_blocks'):
            out = []
            inputs = [P4, P5]
            in_blocks = list(self.in_blocks.values())[5:]
            for i in range(2):
                out.append(in_blocks[i](inputs[i]))
            out += x[-2:]
            x = out
            del inputs, P4, P5, out
        else:
            x = x[1:]
        # x: [P4_02, P5_02, P6_0, P7_0]
        # --------------------------------
        # calculate P4_2, P5_2, P6_2, P7_2
        out_2 = []
        conv_block2 = self.conv_block2
        downsample_block = self.downsample_block
        in_1 = x.copy()  # [P4_02, P5_02, P6_0, P7_0]
        in_2 = list(reversed(out_1))[1:]  # [P4_1, P5_1, P6_1]
        in_3 = [out_1[-1]]  # [P3_2]
        if self.attention:
            weights_2 = [self.to_P4_2_w, self.to_P5_2_w, self.to_P6_2_w, self.to_P7_2_w]
            for i in range(4):
                weight = self.weight_relu(weights_2[i])
                weight = weight / (torch.sum(weight, dim=0) + self.attention_eps)
                if i == 3:  # last
                    out_2.append(conv_block2[i](self.swish(
                        weight[0] * in_1[i] + weight[1] * downsample_block[i](in_3[i]))))
                else:
                    out_2.append(conv_block2[i](self.swish(
                        weight[0] * in_1[i] + weight[1] * in_2[i] + weight[2] * downsample_block[i](in_3[i]))))
                    in_3.append(out_2[-1])
            del weights_2
        else:
            for i in range(4):
                if i == 3:  # last
                    out_2.append(conv_block2[i](self.swish(in_1[i] + downsample_block[i](in_3[i]))))
                else:
                    out_2.append(conv_block2[i](self.swish(in_1[i] + in_2[i] + downsample_block[i](in_3[i]))))
                    in_3.append(out_2[-1])
        del in_1, in_2, in_3
        # output = [P4_2, P5_2, P6_2, P7_2]
        # --------------------------------
        out_2.insert(0, out_1[-1])
        return out_2  # [P3_2, P4_2, P5_2, P6_2, P7_2]


class BiFPN(nn.Sequential):
    def __init__(self, fpn_num_repeat, in_channels_list, fpn_channels, attention, attention_eps=1e-4,
                 bn_momentum=1e-2, bn_eps=1e-3, norm_layer=None):
        norm_layer = norm_layer or nn.BatchNorm2d
        layers = []
        for i in range(fpn_num_repeat):
            layers.append(BiFPNBlock(
                in_channels_list if i == 0 else fpn_channels,
                fpn_channels, attention, attention_eps, bn_momentum, bn_eps, norm_layer
            ))
        super(BiFPN, self).__init__(*layers)
