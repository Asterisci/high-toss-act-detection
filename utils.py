'''
Author: 范国藩
Date: 2021-08-02 10:38:44
Description: 
'''
import cv2

def output_video(args, frame, id):
  out = cv2.VideoWriter(id + '_output.mp4', *args['config'])
  for i in frame:
    out.write(i)
  out.release()

def output_text(id, posList):
  out = {
    'start_id': id, 
    'posList': posList
  }
  print(out)
