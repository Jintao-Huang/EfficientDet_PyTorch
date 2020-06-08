# Author: Jintao Huang
# Time: 2020-6-6

import torch.utils.data as tud
import os
from PIL import Image
import torchvision.transforms as trans
from ..utils import load_from_pickle


def get_dataset_from_pickle(root_dir, pkl_name, images_folder=None, pkl_folder=None, trans_func=None):
    image_fname_list, target_list = \
        load_from_pickle(os.path.join(root_dir, pkl_folder or "pkl", pkl_name))
    return MyDataset(root_dir, image_fname_list, target_list, images_folder, trans_func)


class MyDataset(tud.Dataset):
    def __init__(self, root_dir, image_fname_list, target_list, images_folder=None, trans_func=None):
        """

        :param root_dir: str
        :param image_fname_list: List[str]
        :param target_list: List[Dict]
        :param images_folder: str = "JPEGImages"
        :param trans_func: func (image: PIL.Image, target) -> (image: Tensor[C, H, W] RGB, target)
            默认(self._default_trans_func)
        """
        self.root_dir = root_dir
        self.images_folder = images_folder or "JPEGImages"
        self.image_fname_list = image_fname_list
        self.target_list = target_list
        self.trans_func = trans_func or self._default_trans_func

    def __getitem__(self, idx):
        image_fname = self.image_fname_list[idx]
        target = self.target_list[idx]
        images_dir = os.path.join(self.root_dir, self.images_folder)

        if isinstance(idx, slice):
            return self.__class__(self.root_dir, image_fname, target, self.images_folder, self.trans_func)
        else:
            image_path = os.path.join(images_dir, image_fname)
            with Image.open(image_path) as image:  # type: Image.Image
                image, target = self.trans_func(image, target)
            return image, target

    def __len__(self):
        return len(self.image_fname_list)

    @staticmethod
    def _default_trans_func(image, target):
        if image.mode != "RGB":
            image = image.convert("RGB")
        image = trans.ToTensor()(image)
        return image, target
