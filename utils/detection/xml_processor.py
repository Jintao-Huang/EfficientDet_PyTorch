# Author: Jintao Huang
# Time: 2020-5-24

import os
import torch
import re
import numpy as np
import cv2 as cv
import shutil
from ..display import imread, draw_target_in_image, resize_max
from ..utils import load_from_pickle, save_to_pickle
from PIL import Image
from .utils import hflip_image


class XMLProcessor:
    """$"""

    def __init__(self, root_dir, images_folder=None, annos_folder=None, pkl_folder=None,
                 category=None, labels_map=None, exist_ok=False):
        """

        :param root_dir: str
        :param images_folder: str
        :param annos_folder: str
        :param pkl_folder: str
        :param category: dict[str: int]  可多对一
        :param labels_map: dict[int: str]
        """
        self.root_dir = root_dir
        self.images_dir = os.path.join(root_dir, images_folder or "JPEGImages")
        self.annos_dir = os.path.join(root_dir, annos_folder or "Annotations")
        self.pkl_dir = os.path.join(root_dir, pkl_folder or "pkl")
        os.makedirs(self.pkl_dir, exist_ok=exist_ok)
        assert category and labels_map
        self.category = category
        self.labels_map = labels_map
        self.exist_ok = exist_ok

    def xmls_to_pickle(self, pkl_fname=None):
        """将xml的文件列表转为pickle
        默认检查: (图片文件存在, 每张图片至少一个目标, 目标名在category中.)

        :param pkl_fname: str = "images_targets.pkl".
        :return: None
        """
        pkl_fname = pkl_fname or "images_targets.pkl"
        annos_dir = self.annos_dir
        pkl_path = os.path.join(self.pkl_dir, pkl_fname)
        if not self.exist_ok and os.path.exists(pkl_path):
            raise FileExistsError("%s is exists" % pkl_path)
        # -----------------------------
        image_fname_list = []  # len(N)
        target_list = []  # len(N * dict("boxes": shape(NUMi, 4), "labels": shape(NUMi,))
        xml_fname_list = os.listdir(annos_dir)
        for i, xml_fname in enumerate(xml_fname_list):
            image_fname, target = self._get_data_from_xml(xml_fname)
            image_fname_list.append(image_fname)
            target_list.append(target)
            print("\r>> %d / %d" % (i + 1, len(xml_fname_list)), end="", flush=True)
        print()
        save_to_pickle((image_fname_list, target_list), pkl_path)
        print("-------------------------------------------------")
        print("Original:")
        self.test_dataset(pkl_fname)

    def calc_anchor_distribute(self, pkl_fname, ratios_div_lines=None, sizes_div_lines=None):
        """查看boxes的比例分布(H / W), 大小分布(size)

        :param pkl_fname: str
        :param ratios_div_lines: Tensor = np.linspace(0, 3, 31).'
        :param sizes_div_lines: Tensor = np.array([0, 8, 16, 32, 64, 128, 256, 512, 1024])
        """
        if ratios_div_lines is None:
            ratios_div_lines = np.linspace(0, 3, 31)
        if sizes_div_lines is None:
            sizes_div_lines = np.array([0, 8, 16, 32, 64, 128, 256, 512, 1024], dtype=np.long)

        pkl_path = os.path.join(self.pkl_dir, pkl_fname)
        _, target_list = load_from_pickle(pkl_path)

        def get_ratio_size(box):
            """获得ratio

            :param box: shape(4,). ltrb
            :return: float
            """
            l, t, r, b = box
            w, h = r - l, b - t
            return (h / w).item(), torch.sqrt(w * h).item()

        def get_distribute_index(arr, x):
            """arr[idx] <= x < arr[idx + 1]"""
            if x < arr[0]:
                raise ValueError("x(%.2f) < arr[0](%.2f)" % (x, arr[0]))
            for idx in reversed(range(len(arr))):
                if x >= arr[idx]:
                    break
            return idx

        # ----------------------------- 计算distribute
        ratios_distribute = np.zeros_like(ratios_div_lines, dtype=np.long)
        sizes_distribute = np.zeros_like(sizes_div_lines, dtype=np.long)
        for i, target in enumerate(target_list):
            for box in target["boxes"]:
                ratio, size = get_ratio_size(box)
                ratio_idx = get_distribute_index(ratios_div_lines, ratio)
                size_idx = get_distribute_index(sizes_div_lines, size)
                ratios_distribute[ratio_idx] += 1
                sizes_distribute[size_idx] += 1

        print("Anchor ratios distribute(floor):")
        for line in ratios_div_lines:
            print("%-7.2f|" % line, end="")
        print()
        for num in ratios_distribute:
            print("%-7d|" % num, end="")
        print()
        print("Anchor sizes distribute(floor):")
        for line in sizes_div_lines:
            print("%-7d|" % line, end="")
        print()
        for num in sizes_distribute:
            print("%-7d|" % num, end="")
        print()

    def test_dataset(self, pkl_fname):
        """测试pickle文件(图片存在, 输出总图片数、各个分类的目标数). 并打印检查信息

        :return: None
        """

        labels_map = self.labels_map
        # --------------------------------
        pkl_path = os.path.join(self.pkl_dir, pkl_fname)
        image_fname_list, target_list = load_from_pickle(pkl_path)

        print("images数量: %d" % len(image_fname_list))
        print("targets数量: %d" % len(target_list))
        # 获取target各个类的数目
        # 1. 初始化classes_num_dict
        classes_num_dict = {label_name: 0 for label_name in labels_map.values()}
        # 2. 累加
        for target in target_list:  # 遍历每一张图片
            for label in target["labels"]:
                label = label.item()
                classes_num_dict[labels_map[label]] += 1
        # 3. 打印
        print("classes_num:")
        for object_name, value in classes_num_dict.items():
            print("\t%s: %d" % (object_name, value))
        print("\tAll: %d" % sum(classes_num_dict.values()))

    def show_dataset(self, pkl_fname, colors_map=None, random=False):
        """展示数据集，一张张展示

        :param pkl_fname: str
        :param colors_map: Dict / List
        :param random: bool
        :return: None
        """

        images_dir = self.images_dir
        labels_map = self.labels_map
        pkl_path = os.path.join(self.pkl_dir, pkl_fname)
        # --------------------
        image_fname_list, target_list = load_from_pickle(pkl_path)
        if random:
            orders = np.random.permutation(range(len(image_fname_list)))
        else:
            orders = range(len(image_fname_list))
        for i in orders:  # 随机打乱
            # 1. 数据结构
            img_fname = image_fname_list[i]
            target = target_list[i]
            img_path = os.path.join(images_dir, img_fname)
            # 2. 打开图片
            image = imread(img_path)
            draw_target_in_image(image, target, colors_map, labels_map)

            image = resize_max(image, 720, 1080)
            cv.imshow("%s" % img_fname, image)
            cv.waitKey(0)
            cv.destroyWindow("%s" % img_fname)

    def concat_pickle(self, processor_list, old_pkl_fname_list, new_pkl_fname=None, prefix_list=None):
        """合并pickle. 图片、新pickle会合并到第一个中.  (no$, 未测试, 可能存在bug)

        :param processor_list: List[Processor]. 不包括self
        :param old_pkl_fname_list: List[str]
        :param new_pkl_fname: str = "images_targets_concat.pkl".
        :param prefix_list: List[str]
        """

        # 1. 输入处理
        processor_list.insert(0, self)
        new_pkl_fname = new_pkl_fname or "images_targets_concat.pkl"
        new_pkl_path = os.path.join(self.pkl_dir, new_pkl_fname)
        if not self.exist_ok and os.path.exists(new_pkl_path):
            raise FileExistsError("%s is exists" % new_pkl_path)
        if prefix_list is None:
            prefix_list = []
            for i in range(len(processor_list)):
                prefix_list.append("_" * i)

        # ------------------------------------------
        old_pkl_path_list, old_images_dir_list = [], []
        for i, processor in enumerate(processor_list):
            old_images_dir_list.append(processor.images_dir)
            old_pkl_path_list.append(os.path.join(processor.pkl_dir, old_pkl_fname_list[i]))

        new_images_dir = old_images_dir_list[0]
        new_image_fname_list, new_target_list = [], []
        for old_images_dir, old_pkl_path, prefix in \
                zip(old_images_dir_list, old_pkl_path_list, prefix_list):

            old_image_fname_list, old_target_list = load_from_pickle(old_pkl_path)
            # 1. 修改target_list
            new_target_list += old_target_list
            # 2. 移动图片. 修改image_fname_list
            for i, image_fname in enumerate(old_image_fname_list):
                # 修改image_fname_list
                new_image_fname = prefix + image_fname
                new_image_fname_list.append(new_image_fname)
                # 移动图片
                if prefix != "":
                    old_path = os.path.join(old_images_dir, image_fname)
                    new_path = os.path.join(new_images_dir, new_image_fname)
                    shutil.copyfile(old_path, new_path)
                print("\r>> %d / %d" % (i + 1, len(old_image_fname_list)), end="")
            print()

        # 3. 保存
        save_to_pickle((new_image_fname_list, new_target_list), new_pkl_path)
        print("-------------------------------------------------")
        print("Concat:")
        self.test_dataset(new_pkl_fname)

    def _get_data_from_xml(self, xml_fname):
        """get img_fname, target from xml. 并检测图片已经存在

        :param xml_fname: str
        :return: tuple(img_fname, target: dict("boxes": Tensor[NUM, 4], "labels": Tensor[NUM]))"""
        images_dir = self.images_dir
        annos_dir = self.annos_dir
        category = self.category
        # 1. 获取文件名
        image_fname = xml_fname.replace(".xml", ".jpg")
        image_path = os.path.join(images_dir, image_fname)
        anno_path = os.path.join(annos_dir, xml_fname)
        # 2. 检测图片存在
        if not os.path.exists(image_path):  # 图片不存在
            raise FileNotFoundError("%s not found" % image_path)

        # 3. 获取ann数据
        with open(anno_path, "r", encoding="utf-8") as f:
            text = f.read()
        data_list = re.findall(  # len(NUMi, 5(object_name, left, top, right, bottom))
            r"<name>\s*(\w*?)\s*</name>.*?"
            r"<xmin>\s*(\d*?)\s*</xmin>.*?<ymin>\s*(\d*?)\s*</ymin>.*?"
            r"<xmax>\s*(\d*?)\s*</xmax>.*?<ymax>\s*(\d*?)\s*</ymax>",
            text, re.DOTALL)

        if len(data_list) == 0:  # 没有框
            print("| no target in %s. but we still put it in" % image_fname)
        # 4. 处理数据
        box_list, label_list = [], []  # len(NUMi, 4), len(NUMi)
        for object_name, left, top, right, bottom in data_list:
            label = category.get(object_name)  # object_name 是否存在于 category中. label: int
            if label is None:  # 目标名不在category中
                raise ValueError("`%s` not in category. path: %s" % (object_name, anno_path))
            if label == -1:
                continue
            box_list.append([int(left), int(top), int(right), int(bottom)])  # int() 不需要担心 str存在空格
            label_list.append(label)

        # 5. 数据类型转换
        target = {
            "boxes": torch.tensor(box_list, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(label_list, dtype=torch.long).reshape(-1)
        }

        return image_fname, target

    def split_train_test_from_pickle(self, total_pkl_fname, test_num=1000,
                                     train_pkl_fname=None, test_pkl_fname=None):
        """将pickle分为训练集和测试集

        :param total_pkl_fname: str
        :param test_num: int. 测试集数量(图片张数)
        :param train_pkl_fname: str = "images_targets_train.pkl"
        :param test_pkl_fname: str = "images_targets_test.pkl"
        :return: None
        """

        train_pkl_fname = train_pkl_fname or "images_targets_train.pkl"
        test_pkl_fname = test_pkl_fname or "images_targets_test.pkl"

        total_pkl_path = os.path.join(self.pkl_dir, total_pkl_fname)
        train_pkl_path = os.path.join(self.pkl_dir, train_pkl_fname)
        test_pkl_path = os.path.join(self.pkl_dir, test_pkl_fname)

        if not self.exist_ok and (os.path.exists(train_pkl_path) or os.path.exists(test_pkl_path)):
            raise FileExistsError("%s or %s is exists" % (train_pkl_path, test_pkl_path))

        total_image_fname_list, total_target_list = load_from_pickle(total_pkl_path)

        # 乱序处理
        total_image_fname_list = np.stack(total_image_fname_list, 0)
        total_target_fname_list = np.stack(total_target_list, 0)
        shuffle_order = np.random.permutation(len(total_image_fname_list))
        train_order = shuffle_order[:-test_num]
        test_order = shuffle_order[-test_num:]
        # 3. 分开
        # 训练集
        train_image_fname_list = list(total_image_fname_list[train_order])
        train_target_list = list(total_target_fname_list[train_order])
        # 测试集
        test_image_fname_list = list(total_image_fname_list[test_order])
        test_target_list = list(total_target_fname_list[test_order])

        save_to_pickle((train_image_fname_list, train_target_list), train_pkl_path)
        save_to_pickle((test_image_fname_list, test_target_list), test_pkl_path)
        print("-------------------------------------------------")
        print("Train:")
        self.test_dataset(train_pkl_fname)
        print("-------------------------------------------------")
        print("Test:")
        self.test_dataset(test_pkl_fname)

    def make_mini_dataset(self, total_pkl_fname, dataset_num=1000, mini_pkl_fname=None):
        """制作小数据集

        :param total_pkl_fname: str
        :param dataset_num: int. 数据集数量(图片张数)
        :param mini_pkl_fname: str = "images_targets_mini.pkl"
        :return: None
        """

        mini_pkl_fname = mini_pkl_fname or "images_targets_mini.pkl"

        total_pkl_path = os.path.join(self.pkl_dir, total_pkl_fname)
        mini_pkl_path = os.path.join(self.pkl_dir, mini_pkl_fname)

        if not self.exist_ok and os.path.exists(mini_pkl_path):
            raise FileExistsError("%s is exists" % mini_pkl_path)

        total_image_fname_list, total_target_list = load_from_pickle(total_pkl_path)

        # 乱序处理
        total_image_fname_list = np.stack(total_image_fname_list, 0)
        total_target_fname_list = np.stack(total_target_list, 0)
        shuffle_order = np.random.permutation(len(total_image_fname_list))
        mini_order = shuffle_order[:dataset_num]
        # 3. 分开
        # mini集
        mini_image_fname_list = list(total_image_fname_list[mini_order])
        mini_target_list = list(total_target_fname_list[mini_order])

        save_to_pickle((mini_image_fname_list, mini_target_list), mini_pkl_path)
        print("-------------------------------------------------")
        print("Mini:")
        self.test_dataset(mini_pkl_fname)

    def hflip_from_pickle(self, old_pkl_fname, new_pkl_fname=None, prefix="-"):
        """将images, 以及pickle进行水平翻转


        :param old_pkl_fname: str
        :param new_pkl_fname: str = "images_targets_hflip.pkl".
        :param prefix: str. 加上前缀
        :return: None
        """
        new_pkl_fname = new_pkl_fname or "images_targets_hflip.pkl"
        old_pkl_path = os.path.join(self.pkl_dir, old_pkl_fname)
        new_pkl_path = os.path.join(self.pkl_dir, new_pkl_fname)
        if not self.exist_ok and os.path.exists(new_pkl_path):
            raise FileExistsError("%s is exists" % new_pkl_path)
        image_fname_list, target_list = load_from_pickle(old_pkl_path)  # 直接加入
        image_fname_len = len(image_fname_list)  # 原来的长度
        for i in range(image_fname_len):
            image_fname, target = image_fname_list[i], target_list[i]
            old_path = os.path.join(self.images_dir, image_fname)
            new_image_fname = prefix + image_fname
            new_path = os.path.join(self.images_dir, new_image_fname)
            with Image.open(old_path) as image:
                image, target = hflip_image(image, target)  # 翻转图片
            if not self.exist_ok and os.path.exists(new_path):
                raise FileExistsError("%s is exists" % new_path)
            image.save(new_path)
            image_fname_list.append(new_image_fname)
            target_list.append(target)
            print("\r>> %d / %d" % (i + 1, image_fname_len), end="")
        print()

        # 3. 保存
        save_to_pickle((image_fname_list, target_list), new_pkl_path)
        print("-------------------------------------------------")
        print("HFlip:")
        self.test_dataset(new_pkl_fname)
