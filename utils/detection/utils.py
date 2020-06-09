# Author: Jintao Huang
# Time: 2020-5-18

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

import torch.nn as nn
from collections import OrderedDict
import torch
import torchvision.transforms.transforms as trans
import torch.nn.functional as F
from torchvision.ops.boxes import batched_nms
import torchvision.transforms.functional as transF


def collate_fn(x_y_list):
    """

    :param x_y_list: len(N * (x, y))
    :return: x_list, y_list
    """
    return [list(samples) for samples in zip(*x_y_list)]


def to(images, targets, device):
    """

    :param images: List[Tensor[C, H, W]]
    :param targets: List[Dict] / None
    :param device: str / device
    :return: images: List[Tensor], targets: List[Dict] / None
    """
    for i in range(len(images)):
        if images is not None:
            images[i] = images[i].to(device)
        if targets is not None:
            targets[i]["boxes"] = targets[i]["boxes"].to(device)
            targets[i]["labels"] = targets[i]["labels"].to(device)
    return images, targets


def hflip_image(image, target):
    """水平翻转图片, target
    :param image: PIL.Image.
    :param target: Dict
    :return: image_hflip: PIL.Image, target: Dict"""

    image = transF.hflip(image)
    width = image.width
    boxes = target["boxes"].clone()  # ltrb
    boxes[:, 0], boxes[:, 2] = width - boxes[:, 2], width - boxes[:, 0]
    labels = target['labels'].clone()
    return image, {"boxes": boxes, "labels": labels}


class IntermediateLayerGetter(nn.ModuleDict):
    """提取中间层(get intermediate layer) copy from `torchvision.models._utils`"""

    def __init__(self, model, return_layers):
        """

        :param model:
        :param return_layers: Union[dict[str: str], OrderDict]
        """
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")

        orig_return_layers = return_layers
        return_layers = {k: v for k, v in return_layers.items()}
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:  # If all layers are added, the loop is broken
                break

        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x):
        """

        :param x: Tensor[N, C, H, W]
        :return: OrderDict[str: Tensor[N, Ci, Hi, Wi]]
        """
        out = OrderedDict()
        for name, module in self.named_children():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out


class FrozenBatchNorm2d(nn.Module):
    """copy from `torchvision.ops.misc`"""

    def __init__(self, in_channels, eps=1e-5, *args, **kwargs):  # BN: num_features, eps=1e-5, momentum=0.1
        """`*args, **kwargs` to prevent error"""
        self.eps = eps
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(in_channels))
        self.register_buffer("bias", torch.zeros(in_channels))
        self.register_buffer("running_mean", torch.zeros(in_channels))
        self.register_buffer("running_var", torch.ones(in_channels))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))  # Prevent load errors

    def forward(self, x):
        if x.dim() != 4:
            raise ValueError("expected 4D input (got %dD input)" % x.dim())
        mean, var = self.running_mean, self.running_var
        weight, bias = self.weight, self.bias

        mean, var = mean[:, None, None], var[:, None, None]
        weight, bias = weight[:, None, None], bias[:, None, None]
        return (x - mean) * torch.rsqrt(var + self.eps) * weight + bias


def ltrb_to_cxcywh(boxes, as_tuple=False):
    """(left, top, right, bottom) -> (center_x, center_y, width, height)  (已测试)

    :param boxes: shape(NUM/..., 4). (left, top, right, bottom)
    :param as_tuple: if True:
        return len(4 * shape(NUM/...,)). False: return shape(NUM/..., 4)
    :return: len(4 * shape(NUM,)/(...,)) or shape(NUM/..., 4). (cx, cy, w, h)
    """
    left, top, right, bottom = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
    width, height = right - left, bottom - top
    center_x, center_y = left + width / 2, top + height / 2

    if as_tuple:
        return center_x, center_y, width, height
    else:
        return torch.stack([center_x, center_y, width, height], -1)


def cxcywh_to_ltrb(boxes, as_tuple=False):
    """(center_x, center_y, width, height) -> (left, top, right, bottom)  (已测试)

    :param boxes: shape(NUM/..., 4). (center_x, center_y, width, height)
    :param as_tuple:
        if True: return len(4 * shape(NUM/...,)). False: return shape(NUM/..., 4)
    :return: len(4 * shape(NUM/...,)) or shape(NUM/..., 4). (left, top, right, bottom)
    """
    center_x, center_y, width, height = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
    left, top = center_x - width / 2, center_y - height / 2
    right, bottom = left + width, top + height
    if as_tuple:
        return left, top, right, bottom
    else:
        return torch.stack([left, top, right, bottom], -1)


def encode_boxes(boxes, anchors):
    """boxes, anchors -> reg  (已测试)

    :param boxes: shape(NUM/..., 4). 预测boxes/真实boxes. ltrb
    :param anchors: shape(NUM/..., 4). ltrb
    :return: shape(NUM/..., 4). 回归值(x_reg, y_reg, w_reg, h_reg)

    公式：
        x_reg = (cx_t - cx_a) / w_a
        y_reg = (cy_t - cy_a) / h_a
        w_reg = log(w_t / w_a)
        h_reg = log(h_t / h_a)
    """

    # 1. 计算辅助变量
    cx_t, cy_t, w_t, h_t = ltrb_to_cxcywh(boxes, as_tuple=True)
    cx_a, cy_a, w_a, h_a = ltrb_to_cxcywh(anchors, as_tuple=True)

    # 2. 按公式计算
    x_reg = (cx_t - cx_a) / w_a
    y_reg = (cy_t - cy_a) / h_a
    w_reg = torch.log(w_t / w_a)
    h_reg = torch.log(h_t / h_a)

    # 3. 后处理
    return torch.stack([x_reg, y_reg, w_reg, h_reg], -1)


def decode_boxes(reg, anchors):
    """reg, anchors -> boxes  (已测试)

    :param reg: shape(NUM/..., 4). 回归值(x_reg, y_reg, w_reg, h_reg)
        x_reg, y_reg: 对anchors的中心点进行anchors的w, h比例回归
        w_reg, h_reg: 对anchors进行w, h进行exp^reg缩放
    :param anchors: shape(NUM/..., 4). ltrb
    :return: shape(NUM/..., 4). 预测boxes/真实boxes. ltrb

    公式：
        cx_t = cx_a + x_reg * w_a
        cy_t = cy_a + y_reg * h_a
        w_t = w_a * exp(w_reg)  # w_t > 0
        h_t = h_a * exp(h_reg)  # h_t > 0
    """

    # 1. 计算辅助变量
    cx_a, cy_a, w_a, h_a = ltrb_to_cxcywh(anchors, as_tuple=True)
    x_reg, y_reg, w_reg, h_reg = reg[..., 0], reg[..., 1], reg[..., 2], reg[..., 3]

    # 2. 按公式计算
    cx_t = cx_a + x_reg * w_a
    cy_t = cy_a + y_reg * h_a
    w_t = w_a * torch.exp(w_reg)
    h_t = h_a * torch.exp(h_reg)

    # 3. (cx_t, cy_t, w_t, h_t) -> (left, top, right, bottom)
    return cxcywh_to_ltrb(torch.stack([cx_t, cy_t, w_t, h_t], dim=-1), as_tuple=False)


class ImageList:
    """copy from torchvision.models.detection.image_list"""

    def __init__(self, tensors, image_sizes_ori, image_sizes):
        """
        Arguments:
            tensors (tensor)
            image_sizes_ori (list[tuple(H, W)])
            image_sizes (list[tuple(H, W)]): without padding
        """
        self.tensors = tensors
        self.image_sizes_ori = image_sizes_ori
        self.image_sizes = image_sizes

    def to(self, dtype=None, device=None, *args, **kwargs):
        cast_tensor = self.tensors.to(dtype, device, *args, **kwargs)
        return ImageList(cast_tensor, self.image_sizes_ori, self.image_sizes)


def clip_boxes_to_image(boxes, size):
    """copy from torchvision.ops.boxes

    Clip boxes so that they lie inside an image of size `size`.

    Arguments:
        boxes (Tensor[N, 4]): (left, top, right, bottom)
        size (Tuple(H, W)): size of the image

    Returns:
        clipped_boxes (Tensor[N, 4])
    """
    dim = boxes.dim()
    boxes_x = boxes[..., 0::2]
    boxes_y = boxes[..., 1::2]
    height, width = size
    boxes_x = boxes_x.clamp(min=0, max=width - 1)
    boxes_y = boxes_y.clamp(min=0, max=height - 1)
    clipped_boxes = torch.stack((boxes_x, boxes_y), dim=dim)
    return clipped_boxes.reshape(boxes.shape)


class PreProcess(nn.Module):
    def __init__(self, mean=None, std=None):
        super(PreProcess, self).__init__()
        self.mean = mean or (0.485, 0.456, 0.406)
        self.std = std or (0.229, 0.224, 0.225)
        self.trans_func = trans.Normalize(self.mean, self.std)

    def forward(self, image_list, target_list, max_size):
        """

        :param image_list: List[Tensor[C, H, W]]  const. C=3. RGB
        :param target_list: List[Dict("boxes"...)]  const
        :return: ImageList{.tensor[N, C, H, W]}, targets: List[Dict]"""

        image_sizes_ori = []
        image_sizes = []
        images = []
        targets = []
        for i, image in enumerate(image_list):
            image = self.trans_func(image)
            image_sizes_ori.append(image.shape[-2:])
            image, target = self.resize_max(image, target_list[i] if target_list is not None else None,
                                            max_size, max_size)
            image_sizes.append(image.shape[-2:])
            image = self.zero_padding(image)
            images.append(image)
            if target_list is not None:
                targets.append(target)

        return ImageList(torch.stack(images, dim=0), image_sizes_ori, image_sizes), \
               targets if targets else None

    @staticmethod
    def resize_max(image, target=None, max_width=None, max_height=None):
        """将图像resize成最大最小不超过max_width, max_height的图像

        :param image: Tensor(C, H, W). const
        :param target: dict("boxes"). const
        :param max_width: int
        :param max_height: int
        :return: shape(C, H, W).
        """
        # 1. 输入
        height_ori, width_ori = image.shape[-2:]  # original
        max_width = max_width or width_ori
        max_height = max_height or height_ori
        # 2. 算法
        width = max_width
        height = width / width_ori * height_ori
        if height > max_height:
            height = max_height
            width = height / height_ori * width_ori
        height, width = int(height), int(width)
        image = F.interpolate(image[None], (height, width), mode='bilinear', align_corners=False)[0]
        if target is not None:
            target = {
                "labels": target["labels"],  # 可以clone()
                "boxes": target["boxes"].clone()
            }
            target["boxes"] = target["boxes"] * height / height_ori
        return image, target

    @staticmethod
    def zero_padding(image, padding_value=0., max_padding=None):
        """

        :param image: Tensor[C, H, W]
        :param padding_value: float.
        :param max_padding: int. padding后的最大尺度. 默认max(image.shape)
        :return: Tensor[C, X, X]."""

        max_padding = max_padding or max(image.shape[-2:])
        output = torch.full((image.shape[0], max_padding, max_padding), padding_value,
                            dtype=image.dtype, device=image.device)
        output[:, :image.shape[-2], :image.shape[-1]] = image
        return output


class PostProcess(nn.Module):
    def __init__(self):
        super(PostProcess, self).__init__()

    def forward(self, image_list, classifications, regressions, anchors, score_thresh, nms_thresh):
        batch_size = classifications.shape[0]
        image_sizes = image_list.image_sizes
        image_sizes_ori = image_list.image_sizes_ori
        targets = []
        for i in range(batch_size):
            image_size = image_sizes[i]
            image_size_ori = image_sizes_ori[i]
            classification, regression = classifications[i], regressions[i]
            scores, labels = torch.max(classification, dim=-1)  # 一个anchor只能对应一个分类
            positive_idx = torch.nonzero(scores >= score_thresh, as_tuple=True)
            labels, scores = labels[positive_idx], scores[positive_idx]
            boxes = decode_boxes(regression[positive_idx], anchors[positive_idx])
            boxes = clip_boxes_to_image(boxes, image_size)  # 裁剪超出图片的boxes (Cut out boxes beyond the picture)
            keep_idxs = batched_nms(boxes, scores, labels, nms_thresh)  # scores 不用预先排序. 输出的内容经排序
            scores, labels, boxes = scores[keep_idxs], labels[keep_idxs], boxes[keep_idxs]
            boxes *= image_size_ori[0] / image_size[0]  # boxes回归原图
            targets.append({
                "labels": labels,
                "boxes": boxes,
                "scores": scores
            })
        return targets
