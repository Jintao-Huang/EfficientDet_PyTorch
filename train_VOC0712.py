# Author: Jintao Huang
# Time: 2020-6-7
# 此程序直接运行即可，不需要make_dataset.py. (This program can be run directly without make_dataset.py)
# 但请事先下载好数据库并放置到响应位置. (But please download the database in advance and place it in a response location)
import torch
from models.efficientdet import efficientdet_d1
from utils.detection import Trainer, Logger, Tester, Checker, APCounter, Saver, LRScheduler, VOC_Dataset
from tensorboardX import SummaryWriter

batch_size = 32
comment = "-d1,wd=4e-5,bs=32,lr=0.05"

# --------------------------------
voc_dir = r'..'  # image_dir
labels_map = VOC_Dataset.labels_int2str


# --------------------------------
def lr_func(epoch):
    if 0 <= epoch < 1:
        return 1e-4
    elif 1 <= epoch < 3:
        return 1e-3
    elif 3 <= epoch < 5:
        return 0.01
    elif 5 <= epoch < 10:
        return 0.02
    elif 10 <= epoch < 80:
        return 0.05
    elif 80 <= epoch < 100:
        return 0.02
    elif 100 <= epoch < 110:
        return 5e-3
    elif 110 <= epoch < 115:
        return 1e-3
    elif 115 <= epoch < 120:
        return 1e-4


def main():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model = efficientdet_d1(False, num_classes=len(labels_map))

    optim = torch.optim.SGD(model.parameters(), 0, 0.9, weight_decay=4e-5)
    # 数据集自行合并(The dataset merges by yourself)
    # 链接：https://pan.baidu.com/s/17iop7UBnSGExW64cip-pYw
    # 提取码：sdvx
    train_dataset = VOC_Dataset(voc_dir, "0712", "trainval")
    test_dataset = VOC_Dataset(voc_dir, "0712", "test")
    ap_counter = APCounter(labels_map, 0.5)
    writer = SummaryWriter(comment=comment)
    logger = Logger(50, writer)
    checker = Checker(Tester(model, train_dataset, batch_size, device, ap_counter, 1000),
                      Tester(model, test_dataset, batch_size, device, ap_counter, 1000),
                      Saver(model), logger, 8, 2)
    lr_scheduler = LRScheduler(optim, lr_func)
    trainer = Trainer(model, optim, train_dataset, batch_size, device, lr_scheduler, logger, checker)
    print("配置: %s" % comment, flush=True)
    trainer.train((0, 120))
    writer.close()


if __name__ == "__main__":
    main()