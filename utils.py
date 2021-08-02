import numpy as np

def cal_dis(x, y):
  return np.sqrt(np.sum(np.square(y - x)))

def cal_vel(x0, x):
  return x - x0