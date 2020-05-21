# Author: Jintao Huang
# Time: 2020-5-19
import torch.nn as nn
import torch
from .backbone import EfficientNetBackBoneWithBiFPN
from .anchor import AnchorGenerator
from .classifier_regressor import Classifier, Regressor
from .loss import FocalLoss
from .efficientnet import load_params_by_order
from .utils import load_state_dict_from_url, PreProcess, PostProcess, FrozenBatchNorm2d

model_urls = {
    'efficientdet_d0':
        'https://github.com/Jintao-Huang/EfficientDet_PyTorch/releases/download/1.0/efficientdet-d0.pth',
    'efficientdet_d1':
        'https://github.com/Jintao-Huang/EfficientDet_PyTorch/releases/download/1.0/efficientdet-d1.pth',
    'efficientdet_d2':
        'https://github.com/Jintao-Huang/EfficientDet_PyTorch/releases/download/1.0/efficientdet-d2.pth',
    'efficientdet_d3':
        'https://github.com/Jintao-Huang/EfficientDet_PyTorch/releases/download/1.0/efficientdet-d3.pth',
    'efficientdet_d4':
        'https://github.com/Jintao-Huang/EfficientDet_PyTorch/releases/download/1.0/efficientdet-d4.pth',
    'efficientdet_d5':
        'https://github.com/Jintao-Huang/EfficientDet_PyTorch/releases/download/1.0/efficientdet-d5.pth',
    'efficientdet_d6':
        'https://github.com/Jintao-Huang/EfficientDet_PyTorch/releases/download/1.0/efficientdet-d6.pth',
    'efficientdet_d7':
        'https://github.com/Jintao-Huang/EfficientDet_PyTorch/releases/download/1.0/efficientdet-d7.pth',
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


class EfficientDet(nn.Module):
    def __init__(self, backbone_kwargs, num_classes,
                 regressor_classifier_num_repeat, anchor_base_scale,
                 anchor_scales=None, anchor_aspect_ratios=None, norm_layer=None):
        """please use _efficientdet()"""
        super(EfficientDet, self).__init__()

        norm_layer = norm_layer or nn.BatchNorm2d
        fpn_channels = backbone_kwargs['fpn_channels']
        self.image_size = backbone_kwargs['image_size']
        # (2^(1/3)) ^ (0|1|2)
        anchor_scales = anchor_scales or (1., 2 ** (1 / 3.), 2 ** (2 / 3.))  # scale on a single feature
        anchor_aspect_ratios = anchor_aspect_ratios or ((1., 1.), (0.7, 1.4), (1.4, 0.7))  # H / W
        num_anchor = len(anchor_scales) * len(anchor_aspect_ratios)
        self.preprocess = PreProcess(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        self.backbone = EfficientNetBackBoneWithBiFPN(**backbone_kwargs)
        self.classifier = Classifier(fpn_channels, num_anchor, num_classes, regressor_classifier_num_repeat,
                                     1e-2, 1e-3, norm_layer)
        self.regressor = Regressor(fpn_channels, num_anchor, regressor_classifier_num_repeat,
                                   1e-2, 1e-3, norm_layer)
        self.anchor_gen = AnchorGenerator(anchor_base_scale, anchor_scales, anchor_aspect_ratios, [3, 4, 5, 6, 7])
        self.loss_fn = FocalLoss(alpha=0.25, gamma=2, divide_line=1 / 9)
        self.postprocess = PostProcess()

    def forward(self, image_list, targets=None, score_thresh=0.5, nms_thresh=0.5):
        """

        :param image_list: List[Tensor[C, H, W]]  [0., 1.]
        :param targets: Dict['labels': List[Tensor[NUMi]], 'boxes': List[Tensor[NUMi, 4]]]
            boxes: left, top, right, bottom
        :return: loss: Dict / result: Dict
        """
        assert isinstance(image_list[0], torch.Tensor)
        image_list = self.preprocess(image_list, self.image_size)
        x = image_list.tensors
        features = self.backbone(x)
        classifications = self.classifier(features)
        regressions = self.regressor(features)
        del features
        anchors = self.anchor_gen(x)
        # 预训练模型的顺序 -> 当前模型顺序
        # y_reg, x_reg, h_reg, w_reg -> x_reg, y_reg, w_reg, h_reg
        regressions[..., 0::2], regressions[..., 1::2] = regressions[..., 1::2], regressions[..., 0::2].clone()
        if self.training:
            assert targets is not None, "targets is None"
            loss = self.loss_fn(classifications, regressions, anchors, targets)
            return loss
        else:
            assert targets is None, "targets is not None"
            result = self.postprocess(image_list, classifications, regressions, anchors, score_thresh, nms_thresh)
            return result


def _efficientdet(model_name, pretrained=False, progress=True,
                  num_classes=90, pretrained_backbone=True, image_size=None, norm_layer=None, **kwargs):
    if pretrained is True:
        norm_layer = norm_layer or FrozenBatchNorm2d
    else:
        norm_layer = norm_layer or nn.BatchNorm2d

    if pretrained:
        pretrained_backbone = False

    strict = kwargs.pop("strict", True)
    if image_size:
        kwargs['image_size'] = image_size
    kwargs['pretrained_backbone'] = pretrained_backbone

    config = dict(zip(('image_size', 'backbone_name', 'fpn_channels', 'fpn_num_repeat',
                       "regressor_classifier_num_repeat", "anchor_base_scale"), config_dict[model_name]))
    for key, value in config.items():
        kwargs.setdefault(key, value)

    # generate backbone_kwargs
    backbone_kwargs = dict()
    for key in list(kwargs.keys()):
        if key in ("backbone_name", "pretrained_backbone", "fpn_channels", "fpn_num_repeat", "image_size"):
            backbone_kwargs[key] = kwargs.pop(key)
    backbone_kwargs['fpn_norm_layer'] = kwargs['norm_layer'] = norm_layer
    # create modules
    model = EfficientDet(backbone_kwargs, num_classes, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[model_name], progress=progress)
        load_params_by_order(model, state_dict, strict)
    return model


def efficientdet_d0(pretrained=False, progress=True, num_classes=90, pretrained_backbone=True,
                    image_size=None, norm_layer=None, **kwargs):
    return _efficientdet("efficientdet_d0", pretrained, progress, num_classes, pretrained_backbone,
                         image_size, norm_layer, **kwargs)


def efficientdet_d1(pretrained=False, progress=True, num_classes=90, pretrained_backbone=True,
                    image_size=None, norm_layer=None, **kwargs):
    return _efficientdet("efficientdet_d1", pretrained, progress, num_classes, pretrained_backbone,
                         image_size, norm_layer, **kwargs)


def efficientdet_d2(pretrained=False, progress=True, num_classes=90, pretrained_backbone=True,
                    image_size=None, norm_layer=None, **kwargs):
    return _efficientdet("efficientdet_d2", pretrained, progress, num_classes, pretrained_backbone,
                         image_size, norm_layer, **kwargs)


def efficientdet_d3(pretrained=False, progress=True, num_classes=90, pretrained_backbone=True,
                    image_size=None, norm_layer=None, **kwargs):
    return _efficientdet("efficientdet_d3", pretrained, progress, num_classes, pretrained_backbone,
                         image_size, norm_layer, **kwargs)


def efficientdet_d4(pretrained=False, progress=True, num_classes=90, pretrained_backbone=True,
                    image_size=None, norm_layer=None, **kwargs):
    return _efficientdet("efficientdet_d4", pretrained, progress, num_classes, pretrained_backbone,
                         image_size, norm_layer, **kwargs)


def efficientdet_d5(pretrained=False, progress=True, num_classes=90, pretrained_backbone=True,
                    image_size=None, norm_layer=None, **kwargs):
    return _efficientdet("efficientdet_d5", pretrained, progress, num_classes, pretrained_backbone,
                         image_size, norm_layer, **kwargs)


def efficientdet_d6(pretrained=False, progress=True, num_classes=90, pretrained_backbone=True,
                    image_size=None, norm_layer=None, **kwargs):
    return _efficientdet("efficientdet_d6", pretrained, progress, num_classes, pretrained_backbone,
                         image_size, norm_layer, **kwargs)


def efficientdet_d7(pretrained=False, progress=True, num_classes=90, pretrained_backbone=True,
                    image_size=None, norm_layer=None, **kwargs):
    return _efficientdet("efficientdet_d7", pretrained, progress, num_classes, pretrained_backbone,
                         image_size, norm_layer, **kwargs)
