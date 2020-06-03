# Author: Jintao Huang
# Time: 2020-5-24

import torch.nn as nn

default_config = {
    # backbone
    "pretrained_backbone": True,
    "backbone_norm_layer": nn.BatchNorm2d,
    "backbone_freeze": ("conv_first", "layer1"),  # "layer2"
    # anchor:
    "anchor_scales": (1., 2 ** (1 / 3.), 2 ** (2 / 3.)),  # scales on a single feature
    "anchor_aspect_ratios": ((1., 1.), (0.7, 1.4), (1.4, 0.7)),  # H, W
    # focal loss
    "alpha": 0.25,  # 0.85
    "gamma": 2,
    # other:
    "other_norm_layer": nn.BatchNorm2d,
}

config_dict = {
    # resolution[% 128 == 0], backbone, fpn_channels, fpn_num_repeat, regressor_classifier_num_repeat,
    # anchor_base_scale(anchor_size / stride)(基准尺度)
    'efficientdet_d0': (512, 'efficientnet_b0', 64, 3, 3, 4.),  #
    'efficientdet_d1': (640, 'efficientnet_b1', 88, 4, 3, 4.),  #
    'efficientdet_d2': (768, 'efficientnet_b2', 112, 5, 3, 4.),  #
    'efficientdet_d3': (896, 'efficientnet_b3', 160, 6, 4, 4.),  #
    'efficientdet_d4': (1024, 'efficientnet_b4', 224, 7, 4, 4.),  #
    'efficientdet_d5': (1280, 'efficientnet_b5', 288, 7, 4, 4.),
    'efficientdet_d6': (1280, 'efficientnet_b6', 384, 8, 5, 4.),  #
    'efficientdet_d7': (1536, 'efficientnet_b6', 384, 8, 5, 5.)
}

# 官方配置  official configuration
# config_dict = {
#     # resolution[% 128 == 0], backbone, fpn_channels, fpn_num_repeat, regressor_classifier_num_repeat,
#     # anchor_base_scale(anchor_size / stride)(基准尺度)
#     'efficientdet_d0': (512, 'efficientnet_b0', 64, *2, 3, 4.),  #
#     'efficientdet_d1': (640, 'efficientnet_b1', 88, *3, 3, 4.),  #
#     'efficientdet_d2': (768, 'efficientnet_b2', 112, *4, 3, 4.),  #
#     'efficientdet_d3': (896, 'efficientnet_b3', 160, *5, 4, 4.),  #
#     'efficientdet_d4': (1024, 'efficientnet_b4', 224, *6, 4, 4.),  #
#     'efficientdet_d5': (1280, 'efficientnet_b5', 288, 7, 4, 4.),
#     'efficientdet_d6': (*1408, 'efficientnet_b6', 384, 8, 5, 4.),  #
#     'efficientdet_d7': (1536, 'efficientnet_b6', 384, 8, 5, 5.)
# }
