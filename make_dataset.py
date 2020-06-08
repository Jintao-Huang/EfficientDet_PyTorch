# Author: Jintao Huang
# Time: 2020-5-21


from utils.detect import XMLProcessor

# --------------------------------
root_dir = r'.'
images_folder = 'JPEGImages'
annos_folder = "Annotations"
pkl_folder = 'pkl'
pkl_fname = "images_targets.pkl"
train_pkl_fname = "images_targets_train.pkl"
train_hflip_pkl_fname = "images_targets_train_hflip.pkl"
test_pkl_fname = "images_targets_test.pkl"
category = {
    "lip": -1,  # ignore
    "tongue": 0
}
labels_map = {
    0: "tongue"
}
# --------------------------------
xml_processor = XMLProcessor(root_dir, images_folder, annos_folder, pkl_folder, category, labels_map, True)
xml_processor.xmls_to_pickle(pkl_fname)
xml_processor.split_train_test_from_pickle(pkl_fname, 750, train_pkl_fname, test_pkl_fname)
xml_processor.hflip_from_pickle(train_pkl_fname, train_hflip_pkl_fname)
xml_processor.calc_anchor_ratios_distribute(pkl_fname)
xml_processor.show_dataset(train_hflip_pkl_fname, {0: (0, 255, 255)}, True)

