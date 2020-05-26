# Author: Jintao Huang
# Time: 2020-5-22

from models.efficientdet import efficientdet_d0
from utils.display import pil_to_cv
import torch
from PIL import Image
import torchvision.transforms.transforms as trans


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# read images(假定)
image_fname = "images/1.png"
with Image.open(image_fname) as image:
    image_o = pil_to_cv(image)
    image_1 = trans.ToTensor()(image).to(device)  # 图片1
del image
image_2 = torch.rand(3, 600, 800).to(device)  # 图片2

# target(假定)
targets = [{
    "labels": torch.tensor([0, 1]).to(device),
    "boxes": torch.tensor([[0., 10., 200., 200.], [100., 20., 500., 100.]]).to(device)
}, {
    "labels": torch.tensor([]).to(device).reshape((0,)),
    "boxes": torch.tensor([]).to(device).reshape((0, 4))
}]

# train
model = efficientdet_d0(True).to(device)
optim = torch.optim.Adam(model.parameters(), 5e-4)
for i in range(20):
    optim.zero_grad()
    loss = model([image_1, image_2], targets)
    loss_sum = sum(loss.values())
    loss_sum.backward()
    optim.step()
    print("loss: %s" % loss)
