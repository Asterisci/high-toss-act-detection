def add_blank(img, blank, color):
  img[blank[0]:blank[0]+blank[2], blank[1]:blank[1]+blank[3]] = 0
  return img