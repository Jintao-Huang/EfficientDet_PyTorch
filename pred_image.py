# Author: Jintao Huang
# Time: 2020-5-21

from models.efficientdet import efficientdet_d0
import torch
from utils.detection.predictor import Predictor
from utils.utils import load_params

# -------------------------------- 参数
image_path = "images/1.png"
score_thresh = 0.2
nms_thresh = 0.2
# --------------------------------
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# pred
model = efficientdet_d0(False)
load_params(model, r"./checkpoints/model.pth")
predictor = Predictor(model, device, None, None)
predictor.pred_image_and_save(image_path, None, "max", score_thresh, nms_thresh)
# predictor.pred_image_and_show(image_path, "max", score_thresh, nms_thresh)  # test function
