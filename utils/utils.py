# Author: Jintao Huang
# Time: 2020-5-24

import pickle
import hashlib
import torch
import numpy as np
from torch.backends import cudnn


def save_to_pickle(data, filepath):
    """$"""
    with open(filepath, "wb") as f:
        pickle.dump(data, f)


def load_from_pickle(filepath):
    """$"""
    with open(filepath, "rb") as f:
        obj = pickle.load(f)
    return obj


def calculate_hash(filepath):
    with open(filepath, "rb") as f:
        buffer = f.read()
    sha256 = hashlib.sha256()
    sha256.update(buffer)
    digest = sha256.hexdigest()
    return digest[:8]


def set_seed(seed=0):
    """网络重现"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    # 取消cudnn加速时的省略精度产生的随机性
    cudnn.deterministic = True
    # cudnn.benchmark = True  # if benchmark == True, deterministic will be False


def save_params(model, filepath):
    torch.save(model.state_dict(), filepath)


def load_params(model, filepath, prefix="", drop_layers=(), strict=True):
    """

    :param model: 变
    :param filepath: str
    :param prefix: 在pth的state_dict加上前缀.
    :param drop_layers: 对加完前缀后的pth进行剔除.
    :param strict: bool
    """

    load_state_dict = torch.load(filepath)
    # 1. 加前缀
    if prefix:
        for key in list(load_state_dict.keys()):
            load_state_dict[prefix + key] = load_state_dict.pop(key)
    # 2. drop
    for key in list(load_state_dict.keys()):
        for layer in drop_layers:
            if layer in key:
                load_state_dict.pop(key)
                break
    return model.load_state_dict(load_state_dict, strict)


def load_params_by_order(model, filepath, strict=True):
    """The parameter name of the pre-training model is different from the parameter name of the model"""
    load_state_dict = torch.load(filepath)
    # --------------------- 算法
    load_keys = list(load_state_dict.keys())
    model_keys = list(model.state_dict().keys())
    assert len(load_keys) == len(model_keys)
    # by order
    for load_key, model_key in zip(load_keys, model_keys):
        load_state_dict[model_key] = load_state_dict.pop(load_key)

    return model.load_state_dict(load_state_dict, strict)


def frozen_layers(model, freeze_layers):
    """冻结层"""
    for name, parameter in model.named_parameters():
        for layer in freeze_layers:
            if layer in name:  # 只要含有名字即可
                parameter.requires_grad_(False)
                break
        else:
            parameter.requires_grad_(True)
