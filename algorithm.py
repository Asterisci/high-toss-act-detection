import cv2
import threading
import numpy as np
from preprocess import add_blank

from utils import *

def diff_track(args, camera, T, blur, dis=100, angle=2):
  fgbg = cv2.createBackgroundSubtractorKNN()

  track_point = []
  frame_id = 0
  display_id = 1
  img = []

  while True:
    (grabbed, frame) = camera.read()

    if not grabbed:
      break

    img.append(frame)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if args['blank']:
      gray = add_blank(gray, args['blank'], 0)
    gray = cv2.GaussianBlur(gray, (blur, blur), 0)

    fgmask = fgbg.apply(gray)
    thresh = cv2.threshold(fgmask, args['target'], 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    (cnts, _) = cv2.findContours(thresh.copy(),
                                 cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in cnts:
      if cv2.contourArea(c) < args["min"] or cv2.contourArea(c) > args["max"]:
        continue

      (x, y, w, h) = cv2.boundingRect(c)
      p = np.array([x, y])

      if frame_id % 1 == 0:

        track_point_np = np.array(track_point)
        if track_point_np.size == 0:
          # [x, y, w, h, insert_frame_id, display_count, stop_count]
          track_point.append([x, y, w, h, frame_id, 0, 0, 0])
          continue

        track_dis = p - track_point_np[:, :2]
        _dis = np.sqrt(np.sum(np.square(track_dis), axis=1))

        if np.min(_dis) > dis:
          track_point_np = np.vstack(
              [track_point_np, [x, y, w, h, frame_id, 0, 0, 0]])
          _index = track_point_np.shape[0] - 1
        else:
          _index = np.argmin(_dis)
          track = track_point_np[_index, :]
          v = np.arctan2(track_dis[_index, 1],
                         (track_dis[_index, 0])) * 180 / np.pi
          if ((v > 60 or v < -30) and track_dis[_index, 1] > 0) or track_point_np[_index, 5] > 8:
            if track_point_np[_index, 5] > 5:
              if track_point_np[_index, 7] == 0:
                track_point_np[_index, 7] = display_id
                display_id += 1
                _record_pos = [x, y, w, h, id]
              # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
              # cv2.putText(frame, str(track_point_np[_index, 7]), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (55,255,155), 2)
            track_point_np[_index, 5] += 1
            track_point_np[_index, :4] = np.array([[x, y, w, h]])

        track_point_np[:, 6] += 1
        track_point_np[_index, 6] = 0

        for point in track_point_np[np.logical_and(track_point_np[:, 6] >= 9, track_point_np[:, 5] > 5)]:
          # reverse_track(args, frame_id, fgbg, img, point, str(point[7]), T, blur, dis)
          t = threading.Thread(target=reverse_track, args=(args, frame_id, fgbg, img, point, str(point[7]), T, blur, dis))
          t.start()
          # thread.start_new_thread( reverse_track, (args, frame_id, fgbg, img, point, str(point[7]), T, blur, dis))

        track_point_np = track_point_np[track_point_np[:, 6] < 9]

        track_point = track_point_np.tolist()

    if len(img) > args['img']:
      img.pop(0)

    # out.write(frame)
    # cv2.imshow("Security Feed", frame)
    # cv2.imshow("Mask", gray)

    # key = cv2.waitKey(1) & 0xFF

    # if key == ord("q"):
    #   break

    frame_id += 1


def reverse_track(args, frame_id, fg, img, end_point, point_id, T, blur, dis=5):

  track_point = end_point
  reverse_count = 0

  last = None

  frame_list = []
  pos_list = []

  for rev_frame in reversed(img):

    pos_list.append(list(track_point[:4]))

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

    fgmask_rev = fg.apply(gray)

    y_0 = track_point[1]-50 if track_point[1]-50 > 0 else 0
    y_1 = track_point[1]+track_point[3] + \
        50 if track_point[1]+track_point[3]+50 < h else h
    x_0 = track_point[0]-50 if track_point[0]-50 > 0 else 0
    x_1 = track_point[0]+track_point[2] + \
        50 if track_point[0]+track_point[2]+50 < w else w
    fgmas_rev = fgmask_rev[y_0:y_1, x_0:x_1]

    # print(y_0, y_1, x_0, x_1)

    thresh = cv2.threshold(fgmas_rev, T, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    (cnts, _) = cv2.findContours(thresh.copy(),
                                 cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # cv2.imshow("mask", thresh)
    # key = cv2.waitKey(10) & 0xFF
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

    cv2.rectangle(rev_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(rev_frame, str(point_id), (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (55, 255, 155), 2)

    # out = args['out']
    # out.write(rev_frame)

    frame_list.append(rev_frame)

    # cv2.imshow("Security Feed", rev_frame)
    # key = cv2.waitKey(10) & 0xFF

    # if np.min(_dis) < 10:
    #   reverse_count += 1
    # else:
    #   reverse_count = 0

    last = gray
    frame_id -= 1

  output_video(args, list(reversed(frame_list)), str(point_id))
  output_text(frame_id, list(reversed(pos_list)))

  return track_point
