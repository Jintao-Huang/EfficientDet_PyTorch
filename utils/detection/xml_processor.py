# Author: Jintao Huang
# Time: 2020-5-24

import os
import torch
import numpy as np
import cv2 as cv
from ..display import imread, draw_target_in_image, resize_max
import xml.etree.ElementTree as ET


class XMLProcessor:
    """$"""

    def __init__(self, root_dir, images_folder=None, annos_folder=None, labels=None, image_set_path=None):
        """

        :param root_dir: str
        :param images_folder: str = "JPEGImages"
        :param annos_folder: str = "Annotations"
        :param labels: list[str] or Dict[str: int]  可多对一
        :param image_set_path: str. 相对路径于root_dir的文件名
        """
        self.root_dir = root_dir
        self.images_dir = os.path.join(root_dir, images_folder or "JPEGImages")
        self.annos_dir = os.path.join(root_dir, annos_folder or "Annotations")
        if image_set_path:
            self.image_set_path = os.path.join(root_dir, image_set_path)
        else:
            self.image_set_path = None
        assert labels is not None
        if isinstance(labels, list):
            labels = dict(zip(labels, range(len(labels))))
        self.labels_str2int = labels
        self.labels_int2str = [k for k, v in labels.items() if v >= 0]
        self.image_fname_list = None  # Len[N_图片]
        # Len[N_图片 * Dict("boxes": shape[NUMi, 4], "labels": shape[NUMi,]]. left, top, right, bottom
        self.target_list = None

    def parse_xmls(self):
        """解析xmls. 检查: (图片文件存在, 每张图片至少一个目标, 目标名在labels中.)

        :return: None
        """
        image_fname_list = []  # len(N_图片)
        target_list = []  # len(N_图片 * dict("boxes": shape(NUMi, 4), "labels": shape(NUMi,)). left, top, right, bottom
        if self.image_set_path:
            with open(self.image_set_path, "r") as f:
                xml_fname_list = ["%s.xml" % x.rstrip('\n') for x in f]
        else:
            xml_fname_list = os.listdir(self.annos_dir)
        for i, xml_fname in enumerate(xml_fname_list):
            image_fname, target = self._get_data_from_xml(xml_fname)
            image_fname_list.append(image_fname)
            target_list.append(target)
        self.image_fname_list = image_fname_list
        self.target_list = target_list
        self.test_dataset()

    def _get_data_from_xml(self, xml_fname):
        """get img_fname, target from xml. 并检测图片已经存在

        :param xml_fname: str
        :return: tuple(img_fname, target: dict("boxes": Tensor[NUM, 4], "labels": Tensor[NUM]))"""
        # 1. 获取文件名
        image_fname = xml_fname.replace(".xml", ".jpg")
        image_path = os.path.join(self.images_dir, image_fname)
        anno_path = os.path.join(self.annos_dir, xml_fname)
        # 2. 检测图片存在
        if not os.path.exists(image_path):  # 图片不存在
            raise FileNotFoundError("%s not found" % image_path)

        # 3. 获取ann数据
        with open(anno_path, "r", encoding="utf-8") as f:
            text = f.read()
        # len(NUMi, 5(object_name, left, top, right, bottom))
        data_list = list(zip(
            ET.parse(anno_path).findall(".//object/name"),
            ET.parse(anno_path).findall(".//object/bndbox/xmin"),
            ET.parse(anno_path).findall(".//object/bndbox/ymin"),
            ET.parse(anno_path).findall(".//object/bndbox/xmax"),
            ET.parse(anno_path).findall(".//object/bndbox/ymax"),
        ))
        if len(data_list) == 0:  # 没有框
            print("| no target in %s. but we still put it in" % image_fname)
        # 4. 处理数据
        box_list, label_list = [], []  # len(NUMi, 4), len(NUMi)
        for object_name, left, top, right, bottom in data_list:
            label = self.labels_str2int.get(object_name.text)  # object_name 是否存在于labels中. label: int
            if label is None:  # 目标名不在labels中
                raise ValueError("`%s` not in labels. path: %s" % (object_name, anno_path))
            if label == -1:
                continue
            box_list.append([int(left.text), int(top.text), int(right.text), int(bottom.text)])
            label_list.append(label)

        # 5. 数据类型转换
        target = {
            "boxes": torch.tensor(box_list, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(label_list, dtype=torch.long).reshape(-1)
        }

        return image_fname, target

    def test_dataset(self):
        """测试pickle文件(输出总图片数、各个分类的目标数).

        :return: None
        """
        print("-------------------------------------------------")
        print("images数量: %d" % len(self.image_fname_list))
        print("targets数量: %d" % len(self.target_list))
        # 获取target各个类的数目
        # 1. 初始化classes_num_dict
        classes_num_dict = {label_name: 0 for label_name in self.labels_int2str}
        # 2. 累加
        for target in self.target_list:  # 遍历每一张图片
            for label in target["labels"]:
                label = label.item()
                classes_num_dict[self.labels_int2str[label]] += 1
        # 3. 打印
        print("classes_num:")
        for object_name, value in classes_num_dict.items():
            print("\t%s: %d" % (object_name, value))
        print("\tAll: %d" % sum(classes_num_dict.values()), flush=True)

    def show_dataset(self, colors_map=None, random=False):
        """展示数据集，一张张展示

        :param colors_map: Dict / List
        :param random: bool
        :return: None
        """
        if random:
            orders = np.random.permutation(range(len(self.image_fname_list)))
        else:
            orders = range(len(self.image_fname_list))
        for i in orders:  # 随机打乱
            # 1. 数据结构
            img_fname = self.image_fname_list[i]
            target = self.target_list[i]
            img_path = os.path.join(self.images_dir, img_fname)
            # 2. 打开图片
            image = imread(img_path)
            draw_target_in_image(image, target, colors_map, self.labels_int2str)

            image = resize_max(image, 720, 1080)
            cv.imshow("%s" % img_fname, image)
            cv.waitKey(0)
            cv.destroyWindow("%s" % img_fname)
