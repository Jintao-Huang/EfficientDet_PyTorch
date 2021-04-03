# Author: Jintao Huang
# Time: 2020-5-18
import torchvision.transforms.functional as transF
import torch
import torchvision.transforms as trans


def collate_fn(x_y_list):
    """

    :param x_y_list: len(N * (x, y))
    :return: List[x_list, y_list]
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
    :param image: PIL.Image. const
    :param target: Dict. const
    :return: image_hflip: PIL.Image, target: Dict"""

    image = transF.hflip(image)
    width = image.width
    boxes = target["boxes"].clone()  # ltrb
    boxes[:, 0], boxes[:, 2] = width - boxes[:, 2], width - boxes[:, 0]
    labels = target['labels'].clone()
    return image, {"boxes": boxes, "labels": labels}


def train_transforms(image, target):
    if image.mode != "RGB":
        image = image.convert("RGB")
    if torch.rand(1) < 0.5:
        image, target = hflip_image(image, target)
    image = trans.ToTensor()(image)
    return image, target


def test_transforms(image, target):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = trans.ToTensor()(image)
    return image, target
