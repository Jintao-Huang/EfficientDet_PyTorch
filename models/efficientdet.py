# Author: Jintao Huang
# Time: 2020-5-19
import torch.nn as nn
import torch
from .backbone import EfficientNetBackBoneWithBiFPN
from .anchor import AnchorGenerator
from .classifier_regressor import Classifier, Regressor
from .loss import FocalLoss
from .utils import load_state_dict_from_url, PreProcess, PostProcess
from .default_config import default_config, config_dict

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


class EfficientDet(nn.Module):
    def __init__(self, num_classes, config):
        """please use _efficientdet()"""
        super(EfficientDet, self).__init__()

        self.image_size = config['image_size']
        fpn_channels = config['fpn_channels']
        anchor_scales = config['anchor_scales']
        anchor_aspect_ratios = config['anchor_aspect_ratios']
        anchor_base_scale = config['anchor_base_scale']
        regressor_classifier_num_repeat = config['regressor_classifier_num_repeat']
        other_norm_layer = config['other_norm_layer']

        # (2^(1/3)) ^ (0|1|2)
        anchor_scales = anchor_scales or (1., 2 ** (1 / 3.), 2 ** (2 / 3.))  # scale on a single feature
        anchor_aspect_ratios = anchor_aspect_ratios or ((1., 1.), (0.7, 1.4), (1.4, 0.7))  # H, W
        num_anchor = len(anchor_scales) * len(anchor_aspect_ratios)
        self.preprocess = PreProcess(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        self.backbone = EfficientNetBackBoneWithBiFPN(config)
        self.classifier = Classifier(fpn_channels, num_anchor, num_classes, regressor_classifier_num_repeat,
                                     1e-2, 1e-3, other_norm_layer)
        self.regressor = Regressor(fpn_channels, num_anchor, regressor_classifier_num_repeat,
                                   1e-2, 1e-3, other_norm_layer)
        self.anchor_gen = AnchorGenerator(anchor_base_scale, anchor_scales, anchor_aspect_ratios, [3, 4, 5, 6, 7])
        self.loss_fn = FocalLoss(alpha=0.25, gamma=2, divide_line=1 / 9)
        self.postprocess = PostProcess()

    def forward(self, image_list, targets=None, image_size=None, score_thresh=None, nms_thresh=None):
        """

        :param image_list: List[Tensor[C, H, W]]  [0., 1.]
        :param targets: Dict['labels': List[Tensor[NUMi]], 'boxes': List[Tensor[NUMi, 4]]]
            boxes: left, top, right, bottom
        :param image_size: int. 真实输入图片的大小
        :return: train模式: loss: Dict
                eval模式: result: Dict
        """
        assert isinstance(image_list[0], torch.Tensor)
        image_size = image_size or self.image_size
        # notice: anchor 32 - 812.7. Please adjust the resolution according to the specific situation
        image_size = min(1920, image_size // 128 * 128)  # 需要被128整除
        image_list, targets = self.preprocess(image_list, targets, image_size)
        x = image_list.tensors
        features = self.backbone(x)
        classifications = self.classifier(features)
        regressions = self.regressor(features)
        del features
        anchors = self.anchor_gen(x)
        # 预训练模型的顺序 -> 当前模型顺序
        # y_reg, x_reg, h_reg, w_reg -> x_reg, y_reg, w_reg, h_reg
        regressions[..., 0::2], regressions[..., 1::2] = regressions[..., 1::2], regressions[..., 0::2].clone()
        if targets is not None:
            if score_thresh is not None or nms_thresh is not None:
                print("Warning: no need to transfer score_thresh or nms_thresh")
            loss = self.loss_fn(classifications, regressions, anchors, targets)
            return loss
        else:
            score_thresh = score_thresh or 0.5
            nms_thresh = nms_thresh or 0.5
            result = self.postprocess(image_list, classifications, regressions, anchors, score_thresh, nms_thresh)
            return result


def _efficientdet(model_name, pretrained=False, num_classes=90, config=None):
    if config is None:
        config = default_config
        config.update(dict(zip(('image_size', 'backbone_name', 'fpn_channels', 'fpn_num_repeat',
                                "regressor_classifier_num_repeat", "anchor_base_scale"), config_dict[model_name])))
    if pretrained:
        config['pretrained_backbone'] = False
    # create modules
    model = EfficientDet(num_classes, config)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[model_name])
        model.load_state_dict(state_dict)

    return model


def efficientdet_d0(pretrained=False, num_classes=90, config=None):
    return _efficientdet("efficientdet_d0", pretrained, num_classes, config)


def efficientdet_d1(pretrained=False, num_classes=90, config=None):
    return _efficientdet("efficientdet_d1", pretrained, num_classes, config)


def efficientdet_d2(pretrained=False, num_classes=90, config=None):
    return _efficientdet("efficientdet_d2", pretrained, num_classes, config)


def efficientdet_d3(pretrained=False, num_classes=90, config=None):
    return _efficientdet("efficientdet_d3", pretrained, num_classes, config)


def efficientdet_d4(pretrained=False, num_classes=90, config=None):
    return _efficientdet("efficientdet_d4", pretrained, num_classes, config)


def efficientdet_d5(pretrained=False, num_classes=90, config=None):
    return _efficientdet("efficientdet_d5", pretrained, num_classes, config)


def efficientdet_d6(pretrained=False, num_classes=90, config=None):
    return _efficientdet("efficientdet_d6", pretrained, num_classes, config)


def efficientdet_d7(pretrained=False, num_classes=90, config=None):
    return _efficientdet("efficientdet_d7", pretrained, num_classes, config)
