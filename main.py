'''
Author: 范国藩
Date: 2021-07-26 10:16:47
Description: 
'''
import argparse
import datetime
import time
import cv2

from algorithm import *

ap = argparse.ArgumentParser()
ap.add_argument("--video", help="path to the video file")
ap.add_argument("--min", type=int, default=0, help="minimum area size")
ap.add_argument("--max", type=int, default=500, help="maximum area size")
ap.add_argument("--blur", type=int, default=5, help="blur")
ap.add_argument("--target", type=int, default=5, help="target")
ap.add_argument("--blank", nargs='+', type=int)
args = vars(ap.parse_args())

camera = cv2.VideoCapture(args['video'])
fps = camera.get(cv2.CAP_PROP_FPS)
width, height = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)), int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
args['config'] = [fourcc, fps, (width, height)]

diff_track(args, camera, args['target'], args['blur'])

camera.release()
cv2.destroyAllWindows()