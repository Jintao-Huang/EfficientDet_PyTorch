# Author: Jintao Huang
# Time: 2020-5-20
import torch
import torch.nn as nn
from .utils import encode_boxes
from torchvision.ops import box_iou


def weighted_binary_focal_loss(y_pred, y_true, alpha=0.25, gamma=2, with_logits=False, reduction="mean"):
    """f(x) = -alpha * (1 - x)^a * ln(x) = alpha * (1 - x)^a * CELoss(x)(已测试)

    :param y_pred: shape = (N,) or (...)
    :param y_true: shape = (N,) or (...)
    :param alpha: 负样本与正样本的权重. The weight of the negative sample and the positive sample
        = alpha * positive + (1 - alpha) * negative
    :param with_logits: y_pred是否未经过sigmoid"""

    if reduction == "mean":
        func = torch.mean
    elif reduction == "sum":
        func = torch.sum
    else:
        raise ValueError("reduction should in ('mean', 'sum')")
    if with_logits:
        y_pred = torch.sigmoid(y_pred)
    y_pred = torch.clamp(y_pred, 1e-6, 1 - 1e-6)

    # 前式与后式关于0.5对称(The former and the latter are symmetric about 0.5)
    # y_true 为-1. 即: 既不是正样本、也不是负样本。
    return func((alpha * y_true * -torch.log(y_pred) * (1 - y_pred) ** gamma +
                 (1 - alpha) * (1 - y_true) * -torch.log(1 - y_pred) * y_pred ** gamma) *
                (y_true >= 0).float())


def smooth_l1_loss(y_pred, y_true, divide_line=1.):
    """无论divide_line为多少, 交界处的梯度为1

    :param y_pred: shape(N, num) or (...)
    :param y_true: shape(N, num) or (...)
    :param divide_line: = 分界线
    :return: ()"""

    diff = torch.abs(y_pred - y_true)
    return torch.mean(torch.where(diff < divide_line, 0.5 / divide_line * diff ** 2, diff - 0.5 * divide_line))


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, divide_line=1 / 9):
        """

        :param alpha: focal_loss的alpha
        :param gamma: focal_loss的gamma
        :param divide_line: smooth_l1_loss的分界线
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.divide_line = divide_line

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
                    classification, labels, self.alpha, self.gamma, False, 'sum'))
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
                classification, labels, self.alpha, self.gamma, False, 'sum') /
                                    max(positive_idxs.shape[0], 1))
            # ---------------------------------------- reg_loss
            boxes = boxes_ori[matched][positive_idxs]
            if boxes.shape[0] == 0:
                reg_loss_total.append(torch.tensor(0.).to(device))
                continue
            anchors_pos = anchors[positive_idxs]  # anchors_positive
            reg_true = encode_boxes(boxes, anchors_pos)
            regression = regression[positive_idxs]
            reg_loss_total.append(smooth_l1_loss(regression, reg_true, self.divide_line))

        class_loss = sum(class_loss_total) / len(class_loss_total)
        reg_loss = sum(reg_loss_total) / len(reg_loss_total)
        return {"class_loss": class_loss, "reg_loss": reg_loss}
