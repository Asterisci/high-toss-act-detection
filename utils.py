# coding=UTF-8

import cv2

def output_video(args, frame, id):
  out = cv2.VideoWriter(id + '_output.mp4', *args['config'])
  for i in frame:
    out.write(i)
  out.release()

def output_text(id, pos_list):
  out = {
    'start_id': id, 
    'pos_list': pos_list
  }
  print(out)
