# Author: Jintao Huang
# Time: 2020-5-21

from models.efficientdet import efficientdet_d0
from display.display import draw_target_in_image, pil_to_cv, imwrite
import torch
from PIL import Image
import torchvision.transforms.transforms as trans

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

image_fname = "images/1.png"

# read image
with Image.open(image_fname) as image:
    image_o = pil_to_cv(image)
    image = trans.ToTensor()(image).to(device)

# pred
model = efficientdet_d0(True).to(device)
model.eval()
with torch.no_grad():
    target = model([image], image_size=1920, score_thresh=0.2, nms_thresh=0.2)[0]

# draw
image = draw_target_in_image(image_o, target)
imwrite(image, "images/1_d0.png")
