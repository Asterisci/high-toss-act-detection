import cv2
import numpy as np
from preprocess import add_blank

from utils import *

def diff_track(args, camera, out, T, blur, dis=100, angle=2):
  fgbg = cv2.createBackgroundSubtractorKNN()

  track_point = []
  frame_id = 0
  display_id = 1

  while True:
    (grabbed, frame) = camera.read()

    if not grabbed:
      break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if args['blank']:
      gray = add_blank(gray, args['blank'], 0) #添加时间水印条
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
          track_point.append([x, y, w, h, frame_id, 0, 0, 0]) # [x, y, w, h, insert_frame_id, display_count, stop_count]
          continue
        
        track_dis = p - track_point_np[:, :2]
        _dis = np.sqrt(np.sum(np.square(track_dis), axis=1))

        if np.min(_dis) > dis:
          track_point_np = np.vstack([track_point_np, [x, y, w, h, frame_id, 0, 0, 0]])
          _index = track_point_np.shape[0] - 1
        else:
          _index = np.argmin(_dis)
          track = track_point_np[_index, :]
          v = np.arctan2(track_dis[_index, 1], (track_dis[_index, 0])) * 180 / np.pi
          if ((v > 60 or v < -30) and track_dis[_index, 1] > 0) or track_point_np[_index, 5] > 8:
            if track_point_np[_index, 5] > 5:
              if track_point_np[_index, 7] == 0:
                track_point_np[_index, 7] = display_id
                display_id += 1
                _record_pos = [x, y, w, h, id]
              cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
              cv2.putText(frame, str(track_point_np[_index, 7]), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (55,255,155), 2)
            track_point_np[_index, 5] += 1
            track_point_np[_index, :4]= np.array([[x, y, w, h]])

        track_point_np[:, 6] += 1
        track_point_np[_index, 6] = 0

        track_point_np = track_point_np[track_point_np[:, 6] < 10]

        track_point = track_point_np.tolist()

    out.write(frame)
    cv2.imshow("Security Feed", frame)
    # cv2.imshow("Mask", gray)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
      break

    frame_id += 1
