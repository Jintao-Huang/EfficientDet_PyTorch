# Author: Jintao Huang
# Time: 2020-6-6

import torch
import torchvision.transforms.transforms as trans
from PIL import Image
from .display import pil_to_cv, cv_to_pil, draw_target_in_image, imwrite, resize_max
import os
import cv2 as cv
import time


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


class Predictor:
    """$"""

    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device
        self._pred_video_now = False

    def pred(self, image, image_size="max", score_thresh=0.5, nms_thresh=0.5):
        """

        :param image: Tensor[C, H, W]
        :param image_size:
        :param score_thresh:
        :param nms_thresh:
        :return: target: Dict
        """
        if image_size == "max":
            image_size = max(image.shape[-2:])
        if not self._pred_video_now:
            self.model.eval()
        with torch.no_grad():
            x, _ = to([image], None, self.device)
            target = self.model(x, None, image_size, score_thresh, nms_thresh)[0]
        return target

    def _pred_image(self, image, image_size="max", score_thresh=0.5, nms_thresh=0.5):
        """

        :param image: Tensor[C, H, W]
        :param image_size: None / int / "max"
        :param score_thresh: float
        :param nms_thresh: float
        :return: None
        """

        image_o = pil_to_cv(image)
        image = trans.ToTensor()(image)

        target = self.pred(image, image_size, score_thresh, nms_thresh)
        draw_target_in_image(image_o, target)
        return image_o

    def pred_image_and_show(self, image_path, image_size="max", score_thresh=0.5, nms_thresh=0.5):
        with Image.open(image_path) as image:
            image = self._pred_image(image, image_size, score_thresh, nms_thresh)
        image_show = resize_max(image, 720, 1080)
        cv.imshow("predictor", image_show)
        cv.waitKey(0)

    @staticmethod
    def _default_out_path(in_path):
        """

        :param in_path: str
        :return: return out_path
        """
        path_split = in_path.rsplit('.', 1)
        return path_split[0] + "_out." + path_split[1]

    def pred_image_and_save(self, image_path, image_out_path=None, image_size="max", score_thresh=0.5, nms_thresh=0.5):
        image_out_path = image_out_path or self._default_out_path(image_path)
        with Image.open(image_path) as image:
            image = self._pred_image(image, image_size, score_thresh, nms_thresh)
        imwrite(image, image_out_path)

    def pred_video_and_save(self, video_path, video_out_path=None, image_size="max", score_thresh=0.5, nms_thresh=0.5,
                            exist_ok=False):
        video_out_path = video_out_path or self._default_out_path(video_path)
        if not exist_ok and os.path.exists(video_out_path):
            raise FileExistsError("%s is exists" % video_out_path)

        cap = cv.VideoCapture(video_path)
        out = cv.VideoWriter(video_out_path, cv.VideoWriter_fourcc(*"mp4v"), int(cap.get(cv.CAP_PROP_FPS)),
                             (int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))
        assert cap.isOpened()
        fps = cap.get(cv.CAP_PROP_FPS)
        frame_num = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        print("视频帧率: %s" % fps)
        self.model.eval()
        self._pred_video_now = True
        for i in range(frame_num):
            ret, image = cap.read()  # BGR
            if ret is False:
                break  # 未读到
            start = time.time()
            image = cv_to_pil(image)  # -> PIL.Image
            image = self._pred_image(image, image_size, score_thresh, nms_thresh)
            print("\r>> %d / %d. 处理时间: %f" % (i + 1, frame_num, time.time() - start), end="", flush=True)
            image_show = resize_max(image, 720, 1080)
            cv.imshow("video", image_show)
            if cv.waitKey(1) in (ord('q'), ord('Q')):
                exit(0)
            out.write(image)
        print()
        cap.release()
        out.release()
        self._pred_video_now = False
