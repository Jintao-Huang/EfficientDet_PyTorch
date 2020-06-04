# Author: Jintao Huang
# Time: 2020-5-18

from .efficientnet import _efficientnet
import torch.nn as nn
from .utils import IntermediateLayerGetter
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


class EfficientNetWithBiFPN(nn.Sequential):
    def __init__(self, config):

        backbone_name = config['backbone_name']
        pretrained_backbone = config['pretrained_backbone']
        backbone_norm_layer = config["backbone_norm_layer"]
        image_size = config['image_size']
        backbone_freeze = config['backbone_freeze']
        # -------------------------
        fpn_norm_layer = config["other_norm_layer"]
        fpn_channels = config['fpn_channels']
        fpn_num_repeat = config['fpn_num_repeat']

        # create modules
        backbone = _efficientnet(backbone_name, pretrained_backbone,
                                 norm_layer=backbone_norm_layer, image_size=image_size)
        # freeze layers (自己看效果)进行freeze
        for name, parameter in backbone.named_parameters():
            for freeze_layer in backbone_freeze:
                if freeze_layer in name:
                    parameter.requires_grad_(False)
                    break
            # else:
            #     parameter.requires_grad_(True)

        return_layers = {"layer3": "P3", "layer5": "P4", "layer7": "P5"}  # "layer2": "P2",
        in_channels_list = efficientnet_out_channels[backbone_name]  # bifpn
        super(EfficientNetWithBiFPN, self).__init__(OrderedDict({
            "body": IntermediateLayerGetter(backbone, return_layers),
            "bifpn": BiFPN(fpn_num_repeat, in_channels_list, fpn_channels,
                           attention=True if "b6" not in backbone_name else False,  # d6, d7 use b6
                           norm_layer=fpn_norm_layer)
        }))
