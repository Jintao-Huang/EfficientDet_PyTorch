# Author: Jintao Huang
# Time: 2020-6-6


class LRScheduler:
    def __init__(self, optim, lr_func):
        self.optim = optim
        self.lr_func = lr_func

    def step(self, epoch):
        lr = self.lr_func(epoch)
        self.optim.param_groups[0]['lr'] = lr
        return lr
