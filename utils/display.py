# Author: Jintao Huang
# Time: 2020-5-21

import numpy as np
import cv2 as cv
from PIL import Image


def imwrite(image, filename):
    """cv无法读取中文字符 (CV cannot read Chinese characters)"""
    retval, arr = cv.imencode('.' + filename.rsplit('.', 1)[1], image)  # retval: 是否保存成功
    if retval is True:
        arr.tofile(filename)
    return retval


def imread(filename):
    """cv无法读取中文字符 (CV cannot read Chinese characters).
    此项目中不会用到此函数，写上去只是为了对称美"""
    arr = np.fromfile(filename, dtype=np.uint8)
    return cv.imdecode(arr, -1)


def draw_box(image, box, color):
    """在给定图像上绘制一个方框 (Draws a box on a given image)

    :param image: shape(H, W, C). BGR
    :param box: len(4), (ltrb)
    :param color: len(3). BGR
    """
    image = np.asarray(image, np.uint8)
    box = np.asarray(box, dtype=np.int)
    cv.rectangle(image, (box[0], box[1]), (box[2], box[3]), color, 2, cv.LINE_4)


def draw_text(image, box, text, rect_color):
    """在图像的方框上方绘制文字 (Draw text above the box of the image)

    :param image: shape(H, W, C). BGR
    :param box: len(4), (ltrb)
    :param text: str
    :param rect_color: BGR
    """
    image = np.asarray(image, np.uint8)
    box = np.asarray(box, dtype=np.int)
    cv.rectangle(image, (box[0] - 1, box[1] - 16), (box[0] + len(text) * 9, box[1]), rect_color, -1, cv.LINE_4)
    cv.putText(image, text, (box[0], box[1] - 4), cv.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 1, cv.LINE_8)


def pil_to_cv(img):
    """转PIL.Image到cv (Turn PIL. The Image to CV)

    :param img: PIL.Image. RGB, RGBA, L.
    :return: ndarray. BGR, BGRA, L  (H, W, C{1, 3, 4})
    """
    mode = img.mode
    arr = np.asarray(img)
    if mode == "RGB":
        arr = cv.cvtColor(arr, cv.COLOR_RGB2BGR)
    elif mode == "RGBA":
        arr = cv.cvtColor(arr, cv.COLOR_RGBA2BGRA)
    elif mode in ("L",):
        arr = arr
    else:
        raise ValueError("img.mode nonsupport")
    return arr


def cv_to_pil(arr):
    """转cv到PIL.Image (Turn The Image from CV to PIL)

    :param arr: ndarray. BGR, BGRA, L
    :return: PIL.Image. RGB, RGBA,L
    """

    if arr.ndim == 2:
        pass
    elif arr.ndim == 3:
        arr = cv.cvtColor(arr, cv.COLOR_BGR2RGB)
    else:  # 4
        arr = cv.cvtColor(arr, cv.COLOR_BGRA2RGBA)
    return Image.fromarray(arr)


def draw_target_in_image(image, target, colors_map=None, labels_map=None):
    """画框在image上 (draw boxes and text in image)

    :param image: ndarray[H, W, C]. BGR.
    :param target: dict("boxes", "labels", "). (ltrb)  不变
    :param colors_map: dict[int: str]/List -> tuple(B, G, R)  # [0, 256).
    :param labels_map: dict[int: str]/List. 将int 映射成 类别. 默认: coco_labels_map
    :return: None
    """
    global colors
    global coco_labels_map

    colors_map = colors_map or colors
    labels_map = labels_map or coco_labels_map
    boxes = target['boxes'].round().int().cpu().numpy()
    labels = target['labels'].cpu().numpy()

    scores = target.get('scores')
    if scores is None:
        scores = np.ones_like(labels)
    else:
        scores = scores.cpu().numpy()
    # draw
    for box, label in zip(boxes, labels):
        color = colors_map[label]
        draw_box(image, box, color=color)  # 画方框
    for box, label, score in zip(boxes, labels, scores):  # 防止框把字盖住
        color = colors_map[label]
        label_name = labels_map[label]
        text = "%s %.2f" % (label_name, score)
        draw_text(image, box, text, color)  # 写字


def resize_max(image, max_height=None, max_width=None):
    """将图像resize成最大不超过max_height, max_width的图像

    :param image: shape(H, W, C). BGR. const
    :param max_width: int
    :param max_height: int
    :return: shape(H, W, C). BGR"""

    # 1. 输入
    height, width = image.shape[:2]
    max_width = max_width or width
    max_height = max_height or height
    # 2. 算法
    out_w = max_width
    out_h = height / width * max_width
    if out_h > max_height:
        out_h = max_height
        out_w = width / height * max_height
    image = cv.resize(image, (int(out_w), int(out_h)))
    return image


coco_labels_map = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus',
    6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant',
    11: '', 12: 'stop sign', 13: 'parking meter', 14: 'bench', 15: 'bird',
    16: 'cat', 17: 'dog', 18: 'horse', 19: 'sheep', 20: 'cow',
    21: 'elephant', 22: 'bear', 23: 'zebra', 24: 'giraffe', 25: '',
    26: 'backpack', 27: 'umbrella', 28: '', 29: '', 30: 'handbag',
    31: 'tie', 32: 'suitcase', 33: 'frisbee', 34: 'skis', 35: 'snowboard',
    36: 'sports ball', 37: 'kite', 38: 'baseball bat', 39: 'baseball glove', 40: 'skateboard',
    41: 'surfboard', 42: 'tennis racket', 43: 'bottle', 44: '', 45: 'wine glass',
    46: 'cup', 47: 'fork', 48: 'knife', 49: 'spoon', 50: 'bowl',
    51: 'banana', 52: 'apple', 53: 'sandwich', 54: 'orange', 55: 'broccoli',
    56: 'carrot', 57: 'hot dog', 58: 'pizza', 59: 'donut', 60: 'cake',
    61: 'chair', 62: 'couch', 63: 'potted plant', 64: 'bed', 65: '',
    66: 'dining table', 67: '', 68: '', 69: 'toilet', 70: '',
    71: 'tv', 72: 'laptop', 73: 'mouse', 74: 'remote', 75: 'keyboard',
    76: 'cell phone', 77: 'microwave', 78: 'oven', 79: 'toaster', 80: 'sink',
    81: 'refrigerator', 82: '', 83: 'book', 84: 'clock', 85: 'vase',
    86: 'scissors', 87: 'teddy bear', 88: 'hair drier', 89: 'toothbrush'
}

colors = [
    (0, 252, 124), (0, 255, 127), (255, 255, 0), (220, 245, 245), (255, 255, 240),
    (205, 235, 255), (196, 228, 255), (212, 255, 127), (226, 43, 138), (135, 184, 222),
    (160, 158, 95), (215, 235, 250), (30, 105, 210), (80, 127, 255), (237, 149, 100),
    (220, 248, 255), (60, 20, 220), (255, 255, 0), (139, 139, 0), (11, 134, 184),
    (169, 169, 169), (107, 183, 189), (0, 140, 255), (204, 50, 153), (122, 150, 233),
    (143, 188, 143), (209, 206, 0), (211, 0, 148), (147, 20, 255), (255, 191, 0),
    (255, 144, 30), (34, 34, 178), (240, 250, 255), (34, 139, 34), (255, 0, 255),
    (220, 220, 220), (255, 248, 248), (0, 215, 255), (32, 165, 218), (114, 128, 250),
    (140, 180, 210), (240, 255, 240), (180, 105, 255), (92, 92, 205), (240, 255, 255),
    (140, 230, 240), (250, 230, 230), (245, 240, 255), (255, 248, 240), (205, 250, 255),
    (230, 216, 173), (128, 128, 240), (255, 255, 224), (210, 250, 250), (211, 211, 211),
    (211, 211, 211), (144, 238, 144), (193, 182, 255), (122, 160, 255), (170, 178, 32),
    (250, 206, 135), (153, 136, 119), (153, 136, 119), (222, 196, 176), (224, 255, 255),
    (0, 255, 0), (50, 205, 50), (230, 240, 250), (255, 0, 255), (170, 205, 102),
    (211, 85, 186), (219, 112, 147), (113, 179, 60), (238, 104, 123), (154, 250, 0),
    (204, 209, 72), (133, 21, 199), (250, 255, 245), (225, 228, 255), (181, 228, 255),
    (173, 222, 255), (230, 245, 253), (0, 128, 128), (35, 142, 107), (0, 165, 255),
    (0, 69, 255), (214, 112, 218), (170, 232, 238), (152, 251, 152), (238, 238, 175)
]
