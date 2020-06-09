# Author: Jintao Huang
# Time: 2020-6-7
from .utils import to, collate_fn
from torch.utils.data import DataLoader
import torch


class Tester:
    def __init__(self, model, test_dataset, batch_size, device, ap_counter, test_samples=1000):
        self.model = model.to(device)
        self.test_loader = DataLoader(test_dataset, batch_size, True, collate_fn=collate_fn, pin_memory=True)
        self.device = device
        self.num_samples = len(test_dataset)
        self.batch_size = batch_size
        self.ap_counter = ap_counter
        self.test_step = test_samples // batch_size

    def test(self, last=False):
        self.model.eval()
        self.ap_counter.init_table()
        with torch.no_grad():
            for i, (x, y) in enumerate(self.test_loader):
                x, y = to(x, y, self.device)
                pred = self.model(x)
                self.ap_counter.add(pred, y)
                if not last and i + 1 == self.test_step:
                    break

            ap_dict = self.ap_counter.get_ap_dict()
            self.ap_counter.init_table()  # clear memory
            self.print_mes(i + 1, ap_dict)
        return ap_dict

    def print_mes(self, steps, ap_dict):
        test_num_samples = min(steps * self.batch_size, self.num_samples)
        print("Test | Samples: %d/%d (%.2f%%)" %
              (test_num_samples, self.num_samples,
               test_num_samples / self.num_samples * 100))
        self.ap_counter.print_ap(ap_dict)
