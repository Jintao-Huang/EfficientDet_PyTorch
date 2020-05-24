# Author: Jintao Huang
# Time: 2020-5-18

from .efficientnet import _efficientnet
import torch.nn as nn
from .utils import IntermediateLayerGetter, FrozenBatchNorm2d
from .bifpn import BiFPN
from collections import OrderedDict

efficientnet_out_channels = {
    # the out_channels of P3/P4/P5.
    "efficientnet_b0": [40, 112, 320],
    "efficientnet_b1": [40, 112, 320],
    "efficientnet_b2": [48, 120, 352],
    "efficientnet_b3": [48, 136, 384],
    "efficientnet_b4": [56, 160, 448],
    "efficientnet_b5": [64, 176, 512],
    "efficientnet_b6": [72, 200, 576],
    "efficientnet_b7": [72, 200, 576]
}


class EfficientNetBackBoneWithBiFPN(nn.Sequential):
    def __init__(self, backbone_name, pretrained_backbone, fpn_channels, fpn_num_repeat, image_size, **kwargs):
        kwargs['image_size'] = image_size
        self.fpn_channels = fpn_channels
        self.image_size = image_size
        fpn_norm_layer = kwargs.pop("fpn_norm_layer", nn.BatchNorm2d)
        # create modules
        if pretrained_backbone:
            backbone_norm_layer = FrozenBatchNorm2d
        else:
            backbone_norm_layer = nn.BatchNorm2d
        backbone = _efficientnet(backbone_name, pretrained_backbone, backbone_norm_layer, **kwargs)
        # freeze layers (自己看效果)进行freeze
        # for name, parameter in backbone.named_parameters():
        #     if 'conv_first' in name or "layer1" in name:  # or "layer2" in name
        #         parameter.requires_grad_(False)

        return_layers = {"layer3": "P3", "layer5": "P4", "layer7": "P5"}  # "layer2": "P2",
        in_channels_list = efficientnet_out_channels[backbone_name]  # bifpn
        super(EfficientNetBackBoneWithBiFPN, self).__init__(OrderedDict({
            "body": IntermediateLayerGetter(backbone, return_layers),
            "bifpn": BiFPN(fpn_num_repeat, in_channels_list, fpn_channels,
                           attention=True if "b6" not in backbone_name else False, norm_layer=fpn_norm_layer)
        }))
