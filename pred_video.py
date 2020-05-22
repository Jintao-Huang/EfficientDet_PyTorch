# Author: Jintao Huang
# Time: 2020-5-22

from models.efficientdet import efficientdet_d0
from display.display import draw_target_in_image, cv_to_pil
import torch
import torchvision.transforms.transforms as trans
import cv2 as cv
import time

# -------------------------- 参数
video_path = "video/1.mp4"
video_out_path = video_path[:-4] + "_out" + video_path[-4:]
score_thresh = 0.35
nms_thresh = 0.5
# --------------------------
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

cap = cv.VideoCapture(video_path)
out = cv.VideoWriter(video_out_path, cv.VideoWriter_fourcc(*"mp4v"), int(cap.get(cv.CAP_PROP_FPS)),
                     (int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))
assert cap.isOpened()
fps = cap.get(cv.CAP_PROP_FPS)
frame_num = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
print("视频帧率: %s" % fps)
model = efficientdet_d0(True).to(device)
model.eval()
for i in range(frame_num):
    # 一帧一帧的捕获  跳着读
    ret, image_o = cap.read()  # BGR
    if ret is False:
        break  # 未读到
    image = cv_to_pil(image_o)  # -> PIL.Image
    image = trans.ToTensor()(image).to(device)  # -> tensor. shape(3, H, W), 0-1
    start = time.time()
    with torch.no_grad():
        target = model([image], image_size=max(image.shape),
                       score_thresh=score_thresh, nms_thresh=nms_thresh)[0]

    print("\r>> %d / %d. 处理时间: %f" % (i + 1, frame_num, time.time() - start), end="")
    new_image = draw_target_in_image(image_o, target)
    out.write(new_image)
print()
cap.release()
out.release()
