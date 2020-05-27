# Author: Jintao Huang
# Time: 2020-5-19
import torch.nn as nn
import torch


class AnchorGenerator(nn.Module):
    def __init__(self, base_scale, scales=None, aspect_ratios=None, pyramid_levels=None):
        """

        :param base_scale: 基准尺度(anchor_size / stride)
        :param scales: tuple[float]. scales in single feature.
        :param aspect_ratios: tuple[tuple(H, W)].
        :param pyramid_levels: tuple[int]
        """
        super(AnchorGenerator, self).__init__()
        self.scales = scales or (1., 2 ** (1 / 3.), 2 ** (2 / 3.))
        aspect_ratios = aspect_ratios or ((1., 1.), (0.7, 1.4), (1.4, 0.7))
        if not isinstance(aspect_ratios[0][0], (list, tuple)):
            self.aspect_ratios = (aspect_ratios,) * len(scales)
        pyramid_levels = pyramid_levels or (3, 4, 5, 6, 7)
        self.strides = [2 ** i for i in pyramid_levels]
        self.base_scale = base_scale
        self.image_size = None  # int
        self.anchors = None

    def forward(self, x):
        """

        :param x: (images)Tensor[N, 3, H, W]. need: {.shape, .device, .dtype}
        :return: anchors[X, 4]  # X -> F*H*W*A. (left, top, right, bottom)
        """
        image_size, dtype, device = x.shape[3], x.dtype, x.device
        if self.image_size == image_size:  # Anchors has been generated
            return self.anchors.to(device, dtype, copy=False)  # default: False
        else:
            self.image_size = image_size

        anchors_all = []
        for stride in self.strides:
            anchors_level = []
            for scale, aspect_ratios in zip(self.scales, self.aspect_ratios):
                for aspect_ratio in aspect_ratios:
                    if image_size % stride != 0:
                        raise ValueError('input size must be divided by the stride.')
                    base_anchor_size = self.base_scale * stride * scale
                    # anchor_h / anchor_w = aspect_ratio
                    anchor_h = base_anchor_size * aspect_ratio[0]
                    anchor_w = base_anchor_size * aspect_ratio[1]
                    shifts_x = torch.arange(stride / 2, image_size, stride, dtype=dtype, device=device)
                    shifts_y = torch.arange(stride / 2, image_size, stride, dtype=dtype, device=device)
                    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
                    shift_x = shift_x.reshape(-1)
                    shift_y = shift_y.reshape(-1)
                    shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)  # (X, 4)
                    # left, top, right, bottom. shape(X, 4)
                    anchors = shifts + torch.tensor([-anchor_w / 2, -anchor_h / 2, anchor_w / 2, anchor_h / 2],
                                                    dtype=dtype, device=device)[None]
                    anchors_level.append(anchors)

            anchors_level = torch.stack(anchors_level, dim=1).reshape(-1, 4)  # shape(X, A, 4) -> (-1, 4)
            anchors_all.append(anchors_level)
        self.anchors = torch.cat(anchors_all, dim=0)  # shape(-1, 4)
        return self.anchors
