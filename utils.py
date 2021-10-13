# coding=UTF-8

import cv2
import time

def output_video(args, frame):
  out = cv2.VideoWriter('video.mp4', *args['config'])
  for i in frame:
    out.write(i)
  cv2.imwrite('./first_frame.jpg', frame[0])
  cv2.imwrite('./last_frame.jpg', frame[-1])
  out.release()

def output_key_frame(frame):
  cv2.imwrite('./key_frame.jpg', frame)

def output_text(id, pos_list):
  out = {
    'start_id': id, 
    'pos_list': pos_list
  }
  print(out)


def output_res(args, frame, key):
  output_video(args, frame)
  output_key_frame(key)

  print(int(time.time()))
