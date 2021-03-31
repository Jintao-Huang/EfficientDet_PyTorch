# Author: Jintao Huang
# Time: 2020-5-20
import torch
import torch.nn as nn
from .utils import encode_boxes
from torchvision.ops import box_iou
import torch.nn.functional as F


def weighted_binary_focal_loss(pred, target, alpha=0.25, gamma=2.):
    """f(x) = alpha * (1 - x)^a * -ln(sigmoid(pred))

    :param pred: shape = (N,)
    :param target: shape = (N,)
    :param alpha: float
        The weight of the negative sample and the positive sample. (alpha * positive + (1 - alpha) * negative)
    :param gamma: float
    :return: shape = ()"""

    # target == -1. It's neither a positive sample nor a negative sample.
    return torch.sum(
        torch.where((target == 0.) | (target == -1), torch.tensor(0.),
                    alpha * (1 - pred) ** gamma * target * -torch.log(pred)) +
        torch.where((target == 1.) | (target == -1), torch.tensor(0.),
                    (1 - alpha) * pred ** gamma * (1 - target) * -torch.log(1 - pred)))


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, beta=1 / 9):
        """

        :param alpha: focal_loss的alpha
        :param gamma: focal_loss的gamma
        :param beta: smooth_l1_loss的分界线
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta

    def forward(self, classifications, regressions, anchors, targets):
        """

        :param classifications: Tensor[N, NUM_X, num_classes]. NUM_X: F*H*W*A
        :param regressions: Tensor[N, NUM_X, 4]. NUM_X: F*H*W*A
        :param anchors: Tensor[NUM_X, 4]. NUM_X: F*H*W*A
        :param targets: Dict['labels': List[Tensor[NUMi]], 'boxes': List[Tensor[NUMi, 4]]]
            boxes: left, top, right, bottom
        :return: dict("class_loss", "reg_loss")
        """
        class_loss_total = []
        reg_loss_total = []
        device = anchors.device

        for i, target in enumerate(targets):  # 遍历每一张图片
            labels_ori, boxes_ori = target['labels'], target['boxes']
            classification, regression = classifications[i], regressions[i]
            if labels_ori.shape[0] == 0:  # 空标签图片
                labels = torch.zeros_like(classification, device=device)
                class_loss_total.append(weighted_binary_focal_loss(
                    classification, labels, self.alpha, self.gamma))
                reg_loss_total.append(torch.tensor(0.).to(device))
                continue
            # ---------------------------------------- class_loss
            iou, matched = box_iou(anchors, boxes_ori).max(dim=1)  # 每个anchors只能对应一个boxes
            matched_labels = labels_ori[matched]
            labels = torch.zeros_like(classification, device=device)
            positive_idxs = torch.nonzero(iou >= 0.5)  # 正标签
            ignore_idxs = torch.nonzero((iou >= 0.4) & (iou < 0.5))  # 既不是负样本，也不是正样本 -> -1(忽略)
            labels[positive_idxs, matched_labels[positive_idxs]] = 1
            labels[ignore_idxs, matched_labels[ignore_idxs]] = -1  # 忽略样本
            class_loss_total.append(weighted_binary_focal_loss(
                classification, labels, self.alpha, self.gamma) /
                                    max(positive_idxs.shape[0], 1))
            # ---------------------------------------- reg_loss
            boxes = boxes_ori[matched][positive_idxs]
            if boxes.shape[0] == 0:
                reg_loss_total.append(torch.tensor(0.).to(device))
                continue
            anchors_pos = anchors[positive_idxs]  # anchors_positive
            reg_true = encode_boxes(boxes, anchors_pos)
            regression = regression[positive_idxs]
            reg_loss_total.append(F.smooth_l1_loss(regression, reg_true, beta=self.beta))

        class_loss = sum(class_loss_total) / len(class_loss_total)
        reg_loss = sum(reg_loss_total) / len(reg_loss_total)
        return {"class_loss": class_loss, "reg_loss": reg_loss}
