# coding=UTF-8

import cv2
import threading
import numpy as np
from preprocess import add_blank

from utils import *

def diff_track(args, camera, T, blur):
  cuda_stream = cv2.cuda_Stream()
  fgbg = cv2.cuda.createBackgroundSubtractorMOG2()

  # 检测队列
  track_point = []
  frame_id = 0
  display_id = 1
  img = []
  dis = args['dis']
  anglemax = args['anglemax']
  anglemin = args['anglemin']
  count = args['count']
  updatecount = args['updatecount']
  reversedis = args['reversedis']
  gpu_gray = cv2.cuda_GpuMat()

  while True:
    # 获得帧
    (grabbed, frame) = camera.read()

    # 如果没有帧就停止
    if not grabbed:
      break

    # 回放序列
    img.append(frame)

    # 灰度化
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 时间水印遮挡
    if args['blank']:
      gray = add_blank(gray, args['blank'], 0)

    # 高斯模糊
    gray = cv2.GaussianBlur(gray, (blur, blur), 0)
    gpu_gray.upload(gray)

    fgmask_gpu = fgbg.apply(gpu_gray, -1, cuda_stream)
    thresh_gpu = cv2.cuda.threshold(fgmask_gpu, args['target'], 255, cv2.THRESH_BINARY)[1]
    thresh = thresh_gpu.download()
    thresh = cv2.dilate(thresh, None, iterations=2)

    # 检测识别
    (cnts, _) = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in cnts:
      
      # 判断检测框大小是否符合
      if cv2.contourArea(c) < args["min"] or cv2.contourArea(c) > args["max"]:
        continue

      # 获得检测框位置信息
      (x, y, w, h) = cv2.boundingRect(c)
      p = np.array([x, y])

      # 这个是方便改成三帧法留的接口
      if frame_id % 1 == 0:

        # 如果没有候选框就直接添加
        track_point_np = np.array(track_point)
        if track_point_np.size == 0:
          track_point.append([x, y, w, h, frame_id, 0, 0, 0])
          continue

        # 有候选框尝试匹配

        # 计算位移/速度，由于时间恒定，所以这两个是一个概念
        track_dis = p - track_point_np[:, :2]
        _dis = np.sqrt(np.sum(np.square(track_dis), axis=1))

        # 大于dis的直接变为新候选框
        if np.min(_dis) > dis:
          track_point_np = np.vstack(
              [track_point_np, [x, y, w, h, frame_id, 0, 0, 0]])
          _index = track_point_np.shape[0] - 1
        
        # 不小于dis判断是否能更新匹配到的候选框
        else:
          _index = np.argmin(_dis)
          track = track_point_np[_index, :]
          # 计算速度角
          v = np.arctan2(track_dis[_index, 1],(track_dis[_index, 0])) * 180 / np.pi
          # 更新规则
          if ((v > anglemin or v < anglemax) and track_dis[_index, 1] > 0) or track_point_np[_index, 5] > 8:
            # 识别到的次数大于count
            if track_point_np[_index, 5] > count:
              # 还没有记录过
              if track_point_np[_index, 7] == 0:
                track_point_np[_index, 7] = display_id
                display_id += 1
                # 成为一个识别目标
                _record_pos = [x, y, w, h, id]
            
            # 识别到的次数
            track_point_np[_index, 5] += 1
            track_point_np[_index, :4] = np.array([[x, y, w, h]])

        # 没有被更新计数
        track_point_np[:, 6] += 1
        # 更新的计数重置
        track_point_np[_index, 6] = 0

        # updatecount次没有更新进入且识别到的次数大于count进入回溯识别
        for point in track_point_np[np.logical_and(track_point_np[:, 6] >= updatecount, track_point_np[:, 5] > count)]:
          t = threading.Thread(target=reverse_track, args=(args, frame_id, fgbg, img, point, str(point[7]), T, blur, reversedis))
          t.start()

        # 进入回溯识别的框从列表删除
        track_point_np = track_point_np[track_point_np[:, 6] < updatecount]

        track_point = track_point_np.tolist()

    # 如果回溯队列超长就出队
    if len(img) > args['img']:
      img.pop(0)

    # frame计数
    frame_id += 1

# 回溯识别
def reverse_track(args, frame_id, fg, img, end_point, point_id, T, blur, dis):

  track_point = end_point
  reverse_count = 0

  last = None

  frame_list = []
  pos_list = []

  gpu_gray_rev = cv2.cuda_GpuMat()

  for rev_frame in reversed(img):

    # 位置数组
    pos_list.append(list(track_point[:4]))

    # 检测空置时间
    if reverse_count > 20:
      break

    gray = cv2.cvtColor(rev_frame, cv2.COLOR_BGR2GRAY)
    if args['blank']:
      gray = add_blank(gray, args['blank'], 0)  # 添加时间水印条
    gray = cv2.GaussianBlur(gray, (blur, blur), 0)

    (h, w) = gray.shape

    if last is None:
      last = gray
      frame_id -= 1
      continue

    gpu_gray_rev.upload(gray)
    gpu_fgmask_rev = fg.apply(gpu_gray_rev)

    fgmask_rev = gpu_fgmask_rev.download()

    # 边界检测
    y_0 = track_point[1]-50 if track_point[1]-50 > 0 else 0
    y_1 = track_point[1]+track_point[3] + \
        50 if track_point[1]+track_point[3]+50 < h else h
    x_0 = track_point[0]-50 if track_point[0]-50 > 0 else 0
    x_1 = track_point[0]+track_point[2] + \
        50 if track_point[0]+track_point[2]+50 < w else w
    fgmas_rev = fgmask_rev[y_0:y_1, x_0:x_1]


    thresh = cv2.threshold(fgmas_rev, T, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    (cnts, _) = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detection_list = []

    for c in cnts:
      if cv2.contourArea(c) < args["min"] or cv2.contourArea(c) > args["max"]:
        continue

      (x, y, w, h) = cv2.boundingRect(c)
      x = x - (50 if track_point[0]-50 > 0 else 0) + track_point[0]
      y = y - (50 if track_point[1]-50 > 0 else 0) + track_point[1]
      detection_list.append([x, y, w, h])

    if len(detection_list) == 0:
      reverse_count += 1
      continue
    else:
      reverse_count = 0
    detection_np = np.array(detection_list)
    detection_dis = detection_np[:, :2]
    detection_dis -= track_point[:2]
    _dis = np.sqrt(np.sum(np.square(detection_dis), axis=1))

    _index = np.argmin(_dis)
    detection_dis += track_point[:2]
    track_point = detection_np[_index, :]
    v = np.arctan2(detection_dis[_index, 1],
                   (detection_dis[_index, 0])) * 180 / np.pi

    # 添加检测框与ID
    cv2.rectangle(rev_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(rev_frame, str(point_id), (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (55, 255, 155), 2)

    # 检测结果视频序列
    frame_list.append(rev_frame)

    last = gray
    frame_id -= 1

  frame_list = list(reversed(frame_list))

  output_res(args, frame_list, frame_list[int(len(frame_list)/2)])
  # for debug
  # output_text(frame_id, frame_list)

  return track_point
