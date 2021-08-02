def add_blank(img, x, y, w, h, color):
  img[x:x+w, y:y+h] = 0
  return img