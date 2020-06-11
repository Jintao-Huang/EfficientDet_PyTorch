# Author: Jintao Huang
# Time: 2020-6-6

from .utils import to, collate_fn
from torch.utils.data import DataLoader
import torch


class RuntimeErrorHandler:
    def __init__(self, ignore_num):
        self.ignore_num_ori = self.ignore_num = ignore_num

    def error(self, e):
        if self.ignore_num > 0:
            print(e, flush=True)
            self.ignore_num -= 1
        else:
            raise e

    def init(self):
        self.ignore_num = self.ignore_num_ori


class Trainer:
    def __init__(self, model, optim, train_dataset, batch_size, device,
                 lr_scheduler=None, logger=None, checker=None, runtime_error_handler=None):
        self.model = model.to(device)
        self.optim = optim
        self.train_loader = DataLoader(train_dataset, batch_size, True, collate_fn=collate_fn, pin_memory=True)
        self.device = device
        self.lr_scheduler = lr_scheduler
        self.logger = logger
        assert checker
        self.checker = checker
        self.runtime_error_handler = runtime_error_handler or RuntimeErrorHandler(ignore_num=2)

    def train(self, epoch_range):
        for epoch in range(*epoch_range):
            self.model.train()
            if self.lr_scheduler:
                self.lr_scheduler.step(epoch)
            lr = self.optim.param_groups[0]['lr']
            self.logger.new_epoch(epoch, len(self.train_loader), lr)
            for i, (x, y) in enumerate(self.train_loader):
                try:
                    x, y = to(x, y, self.device)
                    loss = sum(self.model(x, y).values())
                    self.optim.zero_grad()
                    loss.backward()
                    self.optim.step()
                    if self.logger:
                        self.logger.step(loss.item(), i + 1)
                    self.runtime_error_handler.init()
                except RuntimeError as e:
                    x, y, loss = None, None, None
                    torch.cuda.empty_cache()
                    try:
                        self.runtime_error_handler.error(e)
                    except RuntimeError as e:
                        self.checker.saver.save("tmp_epoch%d_step%d" % (epoch, i + 1))
                        raise e

            if self.checker:
                self.checker.step(epoch, last=(epoch == epoch_range[1] - 1))
