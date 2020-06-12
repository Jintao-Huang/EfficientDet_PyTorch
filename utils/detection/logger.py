# Author: Jintao Huang
# Time: 2020-6-7
import time


class Logger:
    def __init__(self, print_steps, writer):
        """Notice: 需要显式的关闭writer. `writer.close()`"""
        self.writer = writer
        self.print_steps = print_steps
        self.steps_each_epoch = None
        # ----------------
        self.epoch = None
        self.lr = None
        self.steps = None
        self.loss_sum = None
        self.epoch_start_time = None
        self.mini_start_time = None

    def new_epoch(self, epoch, steps_each_epoch, lr):
        if self.writer is None:
            raise ValueError("`self.writer` is None")
        self.epoch = epoch
        self.steps_each_epoch = steps_each_epoch
        self.lr = lr
        self.steps = 0
        self.loss_sum = 0.
        self.epoch_start_time = time.time()
        self.mini_start_time = time.time()

    def step(self, loss, steps):
        self.steps = steps
        self.loss_sum += loss
        if self.steps % self.print_steps == 0 or self.steps == self.steps_each_epoch:
            self._print_mes(last=self.steps == self.steps_each_epoch)
        self.log_mes({"loss": loss})

    def log_mes(self, logs):
        for key, value in logs.items():
            self.writer.add_scalar(key, value, self.epoch * self.steps_each_epoch + self.steps)

    def _print_mes(self, last=False):
        loss_mean = self.loss_sum / self.steps
        if last:
            time_ = time.time() - self.epoch_start_time
            print("Total ", end="")
        else:
            time_ = time.time() - self.mini_start_time
        print("Train| Epoch: %d[%d/%d (%.2f%%)]| Loss: %f| Time: %.4f| LR: %g" %
              (self.epoch, self.steps, self.steps_each_epoch, self.steps / self.steps_each_epoch * 100,
               loss_mean, time_, self.lr), flush=True)
        self.mini_start_time = time.time()
