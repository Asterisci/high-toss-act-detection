# coding=UTF-8

import argparse
import datetime
import time
import cv2
import threading

from algorithm import *

def detection(args):
  urls = args['video'].split(' ')

  for url in urls:

    # 获得视频参数
    print(url)
    camera = cv2.VideoCapture(url)
    fps = camera.get(cv2.CAP_PROP_FPS)
    width, height = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)), int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    args['config'] = [fourcc, fps, (width, height)]

    # 多线程调用实现多路同时识别
    t = threading.Thread(target=diff_track, args=(args, camera, args['target'], args['blur']))
    t.start()

# 获得配置
if __name__ == "__main__":
  ap = argparse.ArgumentParser()
  ap.add_argument("--video", type=str, default='', help="video rstp link") # 流地址
  ap.add_argument("--min", type=int, default=0, help="minimum area size") # 最小检测框
  ap.add_argument("--max", type=int, default=500, help="maximum area size") # 最大检测框
  ap.add_argument("--blur", type=int, default=5, help="blur") # 高斯模糊范围
  ap.add_argument("--target", type=int, default=5, help="target") # 二值化阈值
  ap.add_argument("--blank", nargs='+', type=int) # 时间水印遮挡框
  ap.add_argument("--img", type=int, default=300, help="playback img size") # 回放序列保存长度
  ap.add_argument("--dis", type=int, default=100, help="track point combine distance") # 候选框能到匹配的距离
  ap.add_argument("--anglemax", type=int, default=-30, help="max angle") # 小于此角度
  ap.add_argument("--anglemin", type=int, default=60, help="min angle") # 大于此角度
  ap.add_argument("--count", type=int, default=5, help="detection count") # 识别到的次数
  ap.add_argument("--updatecount", type=int, default=9, help="detection count no longer update count") # 识别到框没有再更新计数
  ap.add_argument("--reversedis", type=int, default=100, help="reverse dis") # 回溯检测检测框大小
  args = vars(ap.parse_args())
  detection(args)
