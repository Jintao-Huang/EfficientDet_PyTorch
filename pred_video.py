# Author: Jintao Huang
# Time: 2020-5-22

from models.efficientdet import efficientdet_d0
import torch
from utils.predictor import Predictor

# -------------------------- 参数
video_path = "video/1.mp4"
video_out_path = video_path.rsplit('.', 1)[0] + "_out." + video_path.rsplit('.', 1)[1]  # 懒得命名变量了
score_thresh = 0.35
nms_thresh = 0.5
# --------------------------

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

model = efficientdet_d0(True).to(device)
predictor = Predictor(model, device)
predictor.pred_video_and_save(video_path, video_out_path, "max", score_thresh, nms_thresh, True)
