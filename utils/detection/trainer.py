# Author: Jintao Huang
# Time: 2020-6-6

from .utils import to, collate_fn
from torch.utils.data import DataLoader


class Trainer:
    def __init__(self, model, optim, train_dataset, batch_size, device,
                 lr_scheduler=None, logger=None, checker=None):
        self.model = model.to(device)
        self.optim = optim
        self.train_loader = DataLoader(train_dataset, batch_size, True, collate_fn=collate_fn, pin_memory=True)
        self.device = device
        self.lr_scheduler = lr_scheduler
        self.logger = logger
        self.checker = checker

    def train(self, epoch_range):
        for epoch in range(*epoch_range):
            self.model.train()
            if self.lr_scheduler:
                self.lr_scheduler.step(epoch)
            lr = self.optim.param_groups[0]['lr']
            self.logger.new_epoch(epoch, len(self.train_loader), lr)
            for x, y in self.train_loader:
                x, y = to(x, y, self.device)
                loss = sum(self.model(x, y).values())
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                if self.logger:
                    self.logger.step(loss.item())
            if self.checker:
                self.checker.step(epoch, last=(epoch == epoch_range[1] - 1))
