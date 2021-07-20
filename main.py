# 导入必要的软件包
import argparse
import datetime
import time
import cv2

from algorithm import *

# 创建参数解析器并解析参数
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-i", "--min-area", type=int, default=0, help="minimum area size")
ap.add_argument("-x", "--max-area", type=int, default=10000, help="maximum area size")
args = vars(ap.parse_args())

camera = cv2.VideoCapture(args['video'])
fps = camera.get(cv2.CAP_PROP_FPS)  # 帧数
width, height = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)), int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 宽高
print((width, height))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))

# three_frame_diff(args, camera, out, 15, 21)
MOG(args, camera, out)

# 清理摄像机资源并关闭打开的窗口
camera.release()
out.release()
cv2.destroyAllWindows()