# Author: Jintao Huang
# Time: 2020-6-6

import torch.utils.data as tud
import os
from PIL import Image
import torchvision.transforms as trans
from ..utils import load_from_pickle, save_to_pickle
from .xml_processor import XMLProcessor


def get_dataset_from_pickle(root_dir, pkl_name, images_folder=None, pkl_folder=None, transforms=None):
    image_fname_list, target_list = \
        load_from_pickle(os.path.join(root_dir, pkl_folder or "pkl", pkl_name))
    return MyDataset(root_dir, image_fname_list, target_list, images_folder, transforms)


class MyDataset(tud.Dataset):
    def __init__(self, root_dir, image_fname_list, target_list, images_folder=None, transforms=None):
        """

        :param root_dir: str
        :param image_fname_list: List[str]
        :param target_list: List[Dict]
        :param images_folder: str = "JPEGImages"
        :param transforms: func(image: PIL.Image, target) -> (image: Tensor[C, H, W] RGB, target)
            默认(self._default_trans_func)
        """
        self.root_dir = root_dir
        self.images_folder = images_folder or "JPEGImages"
        assert len(image_fname_list) == len(target_list)
        self.image_fname_list = image_fname_list
        self.target_list = target_list
        self.transforms = transforms or self._default_transforms

    def __getitem__(self, idx):
        image_fname = self.image_fname_list[idx]
        target = self.target_list[idx]
        images_dir = os.path.join(self.root_dir, self.images_folder)

        if isinstance(idx, slice):
            return self.__class__(self.root_dir, image_fname, target, self.images_folder, self.transforms)
        else:
            image_path = os.path.join(images_dir, image_fname)
            with Image.open(image_path) as image:  # type: Image.Image
                image, target = self.transforms(image, target)
            return image, target

    def __len__(self):
        return len(self.image_fname_list)

    @staticmethod
    def _default_transforms(image, target):
        if image.mode != "RGB":
            image = image.convert("RGB")
        image = trans.ToTensor()(image)
        return image, target


class VOC_Dataset(MyDataset):
    labels_str2int = {
        # Person
        "person": 0,
        # Animal
        "bird": 1, "cat": 2, "cow": 3, "dog": 4, "horse": 5, "sheep": 6,
        # Vehicle
        "aeroplane": 7, "bicycle": 8, "boat": 9, "bus": 10, "car": 11, "motorbike": 12, "train": 13,
        # Indoor:
        "bottle": 14, "chair": 15, "diningtable": 16, "pottedplant": 17, "sofa": 18, "tvmonitor": 19
    }
    labels_int2str = list(labels_str2int.keys())

    def __init__(self, root, year, image_set, transforms=None):
        """

        :param root: str. 存放VOCdevkit的文件夹
        :param year: str. e.g. 0712, 2007, 2012
        :param image_set: str{"train", "val", "trainval", "test"}
        :param transforms: func(image: PIL.Image, target) -> (image: Tensor[C, H, W] RGB, target).
            default: self._default_trans_func
        """
        assert os.path.exists(root), "Please download VOC_datasets to this path"
        root_dir = os.path.join(root, "VOCdevkit", "VOC%s" % year)
        pkl_dir = os.path.join(root_dir, "pkl")
        os.makedirs(pkl_dir, exist_ok=True)
        pkl_path = os.path.join(pkl_dir, "voc_%s_%s" % (year, image_set))
        if os.path.exists(pkl_path):
            image_fname_list, target_list = load_from_pickle(pkl_path)
        else:
            xml_processor = XMLProcessor(root_dir, labels=self.labels_str2int,
                                         image_set_path=r"./ImageSets/Main/%s.txt" % image_set)
            xml_processor.parse_xmls()
            save_to_pickle((xml_processor.image_fname_list, xml_processor.target_list), pkl_path)
            image_fname_list, target_list = xml_processor.image_fname_list, xml_processor.target_list
        super(VOC_Dataset, self).__init__(root_dir, image_fname_list, target_list, transforms)
