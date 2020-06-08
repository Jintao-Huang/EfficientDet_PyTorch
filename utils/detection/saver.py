# Author: Jintao Huang
# Time: 2020-6-7
from ..utils import save_params
import os
import time


class Saver:
    def __init__(self, model):
        self.model = model
        self.save_dir = os.path.join("checkpoints", time.strftime("%Y-%m-%d_%H-%M-%S"))
        os.makedirs(self.save_dir, exist_ok=True)

    def save(self, fname):
        save_params(self.model, os.path.join(self.save_dir, fname))
