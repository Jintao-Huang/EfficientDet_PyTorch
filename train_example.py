# Author: Jintao Huang
# Time: 2020-5-22
# This case applies to Debug to view the network architecture

from models.efficientdet import efficientdet_d0
from utils.display import pil_to_cv
import torch
from PIL import Image
import torchvision.transforms.transforms as trans

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# read images(假定assumption)
image_fname = "images/1.png"
with Image.open(image_fname) as image:
    image_o = pil_to_cv(image)
    image_1 = trans.ToTensor()(image).to(device)  # 图片1
del image
image_2 = torch.rand(3, 600, 800).to(device)  # 图片2

# target(假定assumption)
# 由于是随机标签、y_pred 会反复横跳, 此文件只是展示输入输出格式
targets = [{
    "labels": torch.tensor([0, 1]).to(device),
    "boxes": torch.tensor([[50., 50., 800., 800.], [200., 200., 1300., 1300.]]).to(device)
}, {
    "labels": torch.tensor([]).to(device).reshape((0,)),  # 支持空标签图片(Support empty label images)
    "boxes": torch.tensor([]).to(device).reshape((0, 4))
}]
# train
model = efficientdet_d0(False, 2).to(device)
optim = torch.optim.SGD(model.parameters(), 2e-3, 0.9, weight_decay=1e-4, nesterov=True)
for i in range(10):
    optim.zero_grad()
    loss = model([image_1, image_2], targets)
    loss_sum = sum(loss.values())
    loss_sum.backward()
    optim.step()
    print("loss: %s" % loss)
