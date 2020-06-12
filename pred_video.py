# Author: Jintao Huang
# Time: 2020-5-22

from models.efficientdet import efficientdet_d0
import torch
from utils.detection.predictor import Predictor

# -------------------------- 参数
video_path = "video/1.mp4"
score_thresh = 0.35
nms_thresh = 0.5
# --------------------------

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

model = efficientdet_d0(True)
predictor = Predictor(model, device)
predictor.pred_video_and_save(video_path, None, "max", score_thresh, nms_thresh, exist_ok=False, show_on_time=False)
