'''
Author: 范国藩
Date: 2021-07-27 14:13:30
Description: 
'''
def add_blank(img, blank, color):
  img[blank[0]:blank[0]+blank[2], blank[1]:blank[1]+blank[3]] = 0
  return img
