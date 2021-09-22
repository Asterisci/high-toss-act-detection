import argparse
import datetime
import time
import cv2
import threading

from algorithm import *

def detection(url):
  ap = argparse.ArgumentParser()
  ap.add_argument("--video", nargs='+', type=str, default=url, help="video rstp link")
  ap.add_argument("--min", type=int, default=0, help="minimum area size")
  ap.add_argument("--max", type=int, default=500, help="maximum area size")
  ap.add_argument("--blur", type=int, default=5, help="blur")
  ap.add_argument("--target", type=int, default=5, help="target")
  ap.add_argument("--blank", nargs='+', type=int)
  ap.add_argument("--img", type=int, default=300, help="playback img size")
  args = vars(ap.parse_args())

  urls = args['video'].split(' ')

  for url in urls:
    print(url)
    camera = cv2.VideoCapture(url)
    fps = camera.get(cv2.CAP_PROP_FPS)
    width, height = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)), int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    args['config'] = [fourcc, fps, (width, height)]

    t = threading.Thread(target=diff_track, args=(args, camera, args['target'], args['blur']))
    t.start()

  camera.release()
