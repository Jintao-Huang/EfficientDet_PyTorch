# author: Jintao Huang
# date: 2020-5-14

import torch
import torch.nn as nn
from .utils import load_state_dict_from_url
from torch.autograd import Function
import math
import torchvision.transforms.transforms as trans

__all__ = ["preprocess", "EfficientNet", "efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "efficientnet_b3",
           "efficientnet_b4", "efficientnet_b5", "efficientnet_b6", "efficientnet_b7"]

config_dict = {
    # width_ratio, depth_ratio, resolution[% 32 may be not == 0], dropout_rate
    'efficientnet_b0': (1.0, 1.0, 224, 0.2),
    'efficientnet_b1': (1.0, 1.1, 240, 0.2),
    'efficientnet_b2': (1.1, 1.2, 260, 0.3),
    'efficientnet_b3': (1.2, 1.4, 300, 0.3),
    'efficientnet_b4': (1.4, 1.8, 380, 0.4),
    'efficientnet_b5': (1.6, 2.2, 456, 0.4),
    'efficientnet_b6': (1.8, 2.6, 528, 0.5),
    'efficientnet_b7': (2.0, 3.1, 600, 0.5),
    # 'efficientnet_b8': (2.2, 3.6, 672, 0.5),
    # 'efficientnet_l2': (4.3, 5.3, 800, 0.5),
}

model_urls = {
    'efficientnet_b0':
        'https://github.com/Jintao-Huang/EfficientNet_PyTorch/releases/download/1.0/efficientnet_b0-dbe1bd8e.pth',
    'efficientnet_b1':
        'https://github.com/Jintao-Huang/EfficientNet_PyTorch/releases/download/1.0/efficientnet_b1-b14fa8c5.pth',
    'efficientnet_b2':
        'https://github.com/Jintao-Huang/EfficientNet_PyTorch/releases/download/1.0/efficientnet_b2-c9400113.pth',
    'efficientnet_b3':
        'https://github.com/Jintao-Huang/EfficientNet_PyTorch/releases/download/1.0/efficientnet_b3-c3c993c0.pth',
    'efficientnet_b4':
        'https://github.com/Jintao-Huang/EfficientNet_PyTorch/releases/download/1.0/efficientnet_b4-2d749d8b.pth',
    'efficientnet_b5':
        'https://github.com/Jintao-Huang/EfficientNet_PyTorch/releases/download/1.0/efficientnet_b5-b417176a.pth',
    'efficientnet_b6':
        'https://github.com/Jintao-Huang/EfficientNet_PyTorch/releases/download/1.0/efficientnet_b6-0cd8f775.pth',
    'efficientnet_b7':
        'https://github.com/Jintao-Huang/EfficientNet_PyTorch/releases/download/1.0/efficientnet_b7-e9c9b785.pth',
}


def preprocess(images, image_size):
    """预处理(preprocessing)

    :param images: List[PIL.Image]
    :param image_size: int
    :return: shape(N, C, H, W)
    """
    output = []
    trans_func = trans.Compose([
        trans.Resize(image_size),
        trans.ToTensor(),
        trans.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    for image in images:
        if image.mode != "RGB":
            image = image.convert(mode="RGB")
        output.append(trans_func(image))
    return torch.stack(output, dim=0)


def get_same_padding(in_size, kernel_size, stride):
    """'Same 'same' operation with tensorflow
    notice：padding=(0, 1, 0, 1) and padding=(1, 1, 1, 1) are different

    padding=(1, 1, 1, 1):
        out(H, W) = (in + [2 * padding] − kernel_size) // stride + 1

    'same' padding=(0, 1, 0, 1):
        out(H, W) = (in + [2 * padding] − kernel_size) / stride + 1

    :param in_size: Union[int, tuple(in_h, in_w)]
    :param kernel_size: Union[int, tuple(kernel_h, kernel_w)]
    :param stride: Union[int, tuple(stride_h, stride_w)]
    :return: padding: tuple(left, right, top, bottom)
    """
    # 1. 输入处理
    in_h, in_w = (in_size, in_size) if isinstance(in_size, int) else in_size
    kernel_h, kernel_w = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
    stride_h, stride_w = (stride, stride) if isinstance(stride, int) else stride
    out_h, out_w = math.ceil(in_h / stride_h), math.ceil(in_w / stride_w)
    # 2. 计算
    pad_h = max((out_h - 1) * stride_h + kernel_h - in_h, 0)
    pad_w = max((out_w - 1) * stride_w + kernel_w - in_w, 0)

    # 3. 输出
    return pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2


def drop_connect(x, drop_p, training):
    """Throw away the whole InvertedResidual Module"""
    if not training:
        return x

    keep_p = 1 - drop_p
    keep_tensors = torch.floor(keep_p + torch.rand((x.shape[0], 1, 1, 1), dtype=x.dtype, device=x.device))
    return x / keep_p * keep_tensors


class SwishImplement(Function):
    """output = x * sigmoid(x)

    内存更高效、但是运算速度可能会略下降
    Memory is more efficient, but the computing speed may be slightly reduced"""

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x * torch.sigmoid(x)

    @staticmethod
    def backward(ctx, output_grad):
        """d_output / dx = x * sigmoid'(x) + x' + sigmoid(x)"""
        x, = ctx.saved_tensors
        sigmoid_x = torch.sigmoid(x)
        return output_grad * (sigmoid_x * (x * (1 - sigmoid_x) + 1))


class Swish(nn.Module):
    def forward(self, x):
        return SwishImplement().apply(x)


class Conv2dStaticSamePadding(nn.Sequential):
    """Conv using 'same' padding in tensorflow"""

    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, bias,
                 image_size):
        padding = get_same_padding(image_size, kernel_size, stride)
        super(Conv2dStaticSamePadding, self).__init__(
            nn.ZeroPad2d(padding),
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, groups=groups, bias=bias)
        )


class Conv2dBNSwish(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, bias,
                 bn_momentum, bn_eps, image_size, norm_layer):
        super(Conv2dBNSwish, self).__init__(
            Conv2dStaticSamePadding(in_channels, out_channels, kernel_size, stride, groups, bias, image_size),
            norm_layer(out_channels, bn_eps, bn_momentum),
            Swish()
        )


class InvertedResidual(nn.Module):
    """Mobile Inverted Residual Bottleneck Block

    also be called MBConv or MBConvBlock"""

    def __init__(self, in_channels, out_channels, kernel_size, expand_ratio, id_skip, stride, se_ratio,
                 bn_momentum, bn_eps, image_size, drop_connect_rate, norm_layer):
        """
        Params:
            image_size(int): using static same padding, image_size is necessary.
            id_skip(bool): direct connection.
            se_ratio(float): reduce and expand
        """
        super(InvertedResidual, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.id_skip = id_skip
        self.drop_connect_rate = drop_connect_rate

        neck_channels = int(in_channels * expand_ratio)
        if expand_ratio > 1:
            self.expand_conv = Conv2dBNSwish(in_channels, neck_channels, 1, 1,
                                             1, False, bn_momentum, bn_eps, image_size, norm_layer)

        self.depthwise_conv = Conv2dBNSwish(neck_channels, neck_channels, kernel_size, stride,
                                            neck_channels, False, bn_momentum, bn_eps, image_size, norm_layer)
        if (se_ratio is not None) and (0 < se_ratio <= 1):
            se_channels = int(in_channels * se_ratio)
            # a Squeeze and Excitation layer
            self.squeeze_excitation = nn.Sequential(
                Conv2dStaticSamePadding(neck_channels, se_channels, 1, 1, 1, True, image_size),
                Swish(),
                Conv2dStaticSamePadding(se_channels, neck_channels, 1, 1, 1, True, image_size),
            )

        self.pointwise_conv = nn.Sequential(
            Conv2dStaticSamePadding(neck_channels, out_channels, 1, 1, 1, False, image_size),
            norm_layer(out_channels, bn_eps, bn_momentum),
        )

    def forward(self, inputs):
        x = inputs
        if hasattr(self, 'expand_conv'):
            x = self.expand_conv(x)
        x = self.depthwise_conv(x)
        if hasattr(self, 'squeeze_excitation'):
            z = torch.mean(x, dim=(2, 3), keepdim=True)  # AdaptiveAvgPool2d
            z = torch.sigmoid(self.squeeze_excitation(z))
            x = z * x  # se is like a door(sigmoid)
            del z
        x = self.pointwise_conv(x)

        if self.id_skip and self.stride == 1 and self.in_channels == self.out_channels:
            if self.drop_connect_rate:
                x = drop_connect(x, self.drop_connect_rate, training=self.training)  # x may be return zero
            x = x + inputs  # skip connection  if x == 0, x = inputs
        return x


class EfficientNet(nn.Module):
    def __init__(self, num_classes=1000,
                 width_ratio=1.0, depth_ratio=1.0, image_size=224, dropout_rate=0.2,
                 b0_inverted_residual_setting=None,
                 bn_momentum=1e-2, bn_eps=1e-3, channels_divisor=8, min_channels=None, drop_connect_rate=0.2,
                 norm_layer=None):
        super(EfficientNet, self).__init__()
        min_channels = min_channels or channels_divisor
        norm_layer = norm_layer or nn.BatchNorm2d

        if b0_inverted_residual_setting is None:
            # num_repeat, input_channels, output_channels will change.
            b0_inverted_residual_setting = [
                #  kernel_size(depthwise_conv), num_repeat, input_channels(first), output_channels(last),
                #  expand_ratio, .id_skip, stride(first), se_ratio
                [3, 1, 32, 16, 1, True, 1, 0.25],
                [3, 2, 16, 24, 6, True, 2, 0.25],
                [5, 2, 24, 40, 6, True, 2, 0.25],
                [3, 3, 40, 80, 6, True, 2, 0.25],
                [5, 3, 80, 112, 6, True, 1, 0.25],
                [5, 4, 112, 192, 6, True, 2, 0.25],
                [3, 1, 192, 320, 6, True, 1, 0.25]
            ]
        inverted_residual_setting = self._calculate_inverted_residual_setting(
            b0_inverted_residual_setting, width_ratio, depth_ratio, channels_divisor, min_channels)
        self.inverted_residual_setting = inverted_residual_setting
        # calculate total_block_num. Used to calculate the drop_connect_rate
        self.drop_connect_rate = drop_connect_rate
        self.block_idx = 0
        self.total_block_num = 0
        for setting in inverted_residual_setting:
            self.total_block_num += setting[1]

        # create modules
        out_channels = inverted_residual_setting[0][2]
        self.conv_first = Conv2dBNSwish(3, out_channels, 3, 2, 1, False, bn_momentum, bn_eps, image_size, norm_layer)
        for i, setting in enumerate(inverted_residual_setting):
            setattr(self, 'layer%d' % (i + 1), self._make_layers(setting, image_size, bn_momentum, bn_eps, norm_layer))

        in_channels = inverted_residual_setting[-1][3]
        out_channels = in_channels * 4
        self.conv_last = Conv2dBNSwish(in_channels, out_channels, 1, 1, 1, False,
                                       bn_momentum, bn_eps, image_size, norm_layer)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(out_channels, num_classes)

    def forward(self, x):
        x = self.conv_first(x)
        for i in range(len(self.inverted_residual_setting)):
            x = getattr(self, 'layer%d' % (i + 1))(x)
        x = self.conv_last(x)
        x = torch.mean(x, dim=(2, 3))
        x = self.dropout(x)
        x = self.fc(x)
        return x

    @staticmethod
    def _calculate_inverted_residual_setting(b0_inverted_residual_setting, width_ratio, depth_ratio,
                                             channels_divisor, min_channels):
        """change (num_repeat, input_channels, output_channels) through ratio"""
        inverted_residual_setting = b0_inverted_residual_setting.copy()
        for i in range(len(b0_inverted_residual_setting)):
            setting = inverted_residual_setting[i]
            # change input_channels, output_channels (width)  round
            setting[2], setting[3] = setting[2] * width_ratio, setting[3] * width_ratio
            in_channels, out_channels = \
                int(max(min_channels, int(setting[2] + channels_divisor / 2) // channels_divisor * channels_divisor)), \
                int(max(min_channels, int(setting[3] + channels_divisor / 2) // channels_divisor * channels_divisor))
            if in_channels < 0.9 * setting[2]:  # prevent rounding by more than 10%
                in_channels += channels_divisor
            if out_channels < 0.9 * setting[3]:
                out_channels += channels_divisor
            setting[2], setting[3] = in_channels, out_channels
            # change num_repeat (depth)  ceil
            setting[1] = int(math.ceil(setting[1] * depth_ratio))
        return inverted_residual_setting

    def _make_layers(self, setting, image_size, bn_momentum, bn_eps, norm_layer):
        """耦合(Coupling) self.block_idx, self.total_block_num"""
        kernel_size, num_repeat, input_channels, output_channels, expand_ratio, id_skip, stride, se_ratio = setting
        layers = []
        for i in range(num_repeat):
            drop_connect_rate = self.drop_connect_rate * self.block_idx / self.total_block_num
            layers.append(InvertedResidual(
                input_channels if i == 0 else output_channels,
                output_channels, kernel_size, expand_ratio, id_skip,
                stride if i == 0 else 1,
                se_ratio, bn_momentum, bn_eps, image_size, drop_connect_rate, norm_layer
            ))
            self.block_idx += 1
        return nn.Sequential(*layers)


def _efficientnet(model_name, pretrained=False, progress=True, num_classes=1000, norm_layer=None, **kwargs):
    config = dict(zip(('width_ratio', 'depth_ratio', 'image_size', 'dropout_rate'), config_dict[model_name]))
    for key, value in config.items():
        kwargs.setdefault(key, value)

    model = EfficientNet(num_classes, norm_layer=norm_layer, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[model_name], progress=progress)
        model.load_state_dict(state_dict)

    return model


def efficientnet_b0(pretrained=False, progress=True, num_classes=1000, norm_layer=None, **kwargs):
    return _efficientnet("efficientnet_b0", pretrained, progress, num_classes, norm_layer, **kwargs)


def efficientnet_b1(pretrained=False, progress=True, num_classes=1000, norm_layer=None, **kwargs):
    return _efficientnet("efficientnet_b1", pretrained, progress, num_classes, norm_layer, **kwargs)


def efficientnet_b2(pretrained=False, progress=True, num_classes=1000, norm_layer=None, **kwargs):
    return _efficientnet("efficientnet_b2", pretrained, progress, num_classes, norm_layer, **kwargs)


def efficientnet_b3(pretrained=False, progress=True, num_classes=1000, norm_layer=None, **kwargs):
    return _efficientnet("efficientnet_b3", pretrained, progress, num_classes, norm_layer, **kwargs)


def efficientnet_b4(pretrained=False, progress=True, num_classes=1000, norm_layer=None, **kwargs):
    return _efficientnet("efficientnet_b4", pretrained, progress, num_classes, norm_layer, **kwargs)


def efficientnet_b5(pretrained=False, progress=True, num_classes=1000, norm_layer=None, **kwargs):
    return _efficientnet("efficientnet_b5", pretrained, progress, num_classes, norm_layer, **kwargs)


def efficientnet_b6(pretrained=False, progress=True, num_classes=1000, norm_layer=None, **kwargs):
    return _efficientnet("efficientnet_b6", pretrained, progress, num_classes, norm_layer, **kwargs)


def efficientnet_b7(pretrained=False, progress=True, num_classes=1000, norm_layer=None, **kwargs):
    return _efficientnet("efficientnet_b7", pretrained, progress, num_classes, norm_layer, **kwargs)
