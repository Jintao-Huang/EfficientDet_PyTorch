# Author: Jintao Huang
# Time: 2020-5-21

from models.efficientdet import efficientdet_d0
from utils.display import draw_target_in_image, pil_to_cv, imwrite
import torch
from PIL import Image
import torchvision.transforms.transforms as trans

# -------------------------------- 参数
image_path = "images/1.png"
image_out_path = image_path[:-4] + "_out" + image_path[-4:]
score_thresh = 0.2
nms_thresh = 0.2
# --------------------------------
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# read image
with Image.open(image_path) as image:
    image_o = pil_to_cv(image)
    image = trans.ToTensor()(image).to(device)

# pred
model = efficientdet_d0(True).to(device)
model.eval()
with torch.no_grad():
    target = model([image], image_size=max(image.shape), score_thresh=score_thresh, nms_thresh=nms_thresh)[0]

# draw
draw_target_in_image(image_o, target)
imwrite(image_o, image_out_path)
