# Author: Jintao Huang
# Time: 2020-5-21

from models.efficientdet import efficientdet_d0
import torch
from utils.predictor import Predictor

# -------------------------------- 参数
image_path = "images/1.png"
image_out_path = image_path.rsplit('.', 1)[0] + "_out." + image_path.rsplit('.', 1)[1]  # 懒得命名变量了
score_thresh = 0.2
nms_thresh = 0.2
# --------------------------------
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# pred
model = efficientdet_d0(True).to(device)
predictor = Predictor(model, device)
predictor.pred_image_and_save(image_path, image_out_path, "max", score_thresh, nms_thresh)
predictor.pred_image_and_show(image_path, "max", score_thresh, nms_thresh)  # test function
