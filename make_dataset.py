# Author: Jintao Huang
# Time: 2020-5-21

from utils.detection import XMLProcessor
from utils.utils import save_to_pickle, load_from_pickle
import os

# --------------------------------
dataset_dir = r'.'  # 数据集所在文件夹
images_folder = 'JPEGImages'
annos_folder = "Annotations"
pkl_folder = 'pkl'
pkl_fname = "images_targets.pkl"

labels = {
    # -1 -> ignore
    "trash": -1,
    # Person
    "person": 0,
    # Animal
    "bird": 1, "cat": 2, "cow": 3, "dog": 4, "horse": 5, "sheep": 6,
    # Vehicle
    "aeroplane": 7, "bicycle": 8, "boat": 9, "bus": 10, "car": 11, "motorbike": 12, "train": 13,
    # Indoor:
    "bottle": 14, "chair": 15, "diningtable": 16, "pottedplant": 17, "sofa": 18, "tvmonitor": 19
    # ...
}

# colors_map = {  # bgr
#     0: (0, 255, 0),
#     1: (0, 0, 255)
#     # ...
# }
colors_map = None
# --------------------------------
xml_processor = XMLProcessor(dataset_dir, images_folder, annos_folder, labels, None)
xml_processor.parse_xmls()
xml_processor.test_dataset()
pkl_dir = os.path.join(dataset_dir, pkl_folder)
pkl_path = os.path.join(pkl_dir, pkl_fname)
os.makedirs(pkl_dir, exist_ok=True)
save_to_pickle((xml_processor.image_fname_list, xml_processor.target_list), pkl_path)
# xml_processor.image_fname_list, xml_processor.target_list = load_from_pickle(pkl_path)
# xml_processor.test_dataset()
xml_processor.show_dataset(colors_map, random=False)
