import imutils
import cv2
import numpy as np

def frame_diff(args, camera, out, T=10, blur=15):
  lastFrame = None

  while True:
    (grabbed, frame) = camera.read()

    if not grabbed:
      break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (blur, blur), 0)

    if lastFrame is None:
      lastFrame = gray
      continue

    delta = cv2.absdiff(lastFrame, gray)
    thresh = cv2.threshold(delta, T, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    (cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    for c in cnts:
      if cv2.contourArea(c) < args["min_area"] or cv2.contourArea(c) > args["max_area"]:
        continue

      (x, y, w, h) = cv2.boundingRect(c)
      cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    out.write(frame)
    cv2.imshow("Security Feed", frame)
    # cv2.imshow("gray", gray)
    # cv2.imshow("Thresh", thresh)
    # cv2.imshow("Frame Delta", delta)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
      break

    lastFrame = gray

def three_frame_diff(args, camera, out, T=10, blur=15):
  firstFrame = None
  secondFrame = None

  while True:
    (grabbed, frame) = camera.read()

    if not grabbed:
      break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (blur, blur), 0)

    if firstFrame is None:
      firstFrame = gray
      continue
    if secondFrame is None:
      secondFrame = gray
      continue

    delta1 = cv2.absdiff(firstFrame, secondFrame)
    delta2 = cv2.absdiff(secondFrame, gray)
    delta = cv2.bitwise_and(delta1, delta2)
    thresh = cv2.threshold(delta, T, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    (cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    for c in cnts:
      if cv2.contourArea(c) < args["min_area"] or cv2.contourArea(c) > args["max_area"]:
        continue

      (x, y, w, h) = cv2.boundingRect(c)
      cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    out.write(frame)
    cv2.imshow("Security Feed", frame)
    # cv2.imshow("gray", gray)
    # cv2.imshow("Thresh", thresh)
    # cv2.imshow("Frame Delta", delta)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
      break

    firstFrame = secondFrame
    secondFrame = gray

def MOG(args, camera, out, T=10, blur=15):
  fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

  while True:
    (grabbed, frame) = camera.read()

    if not grabbed:
      break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (blur, blur), 0)

    fgmask = fgbg.apply(gray)
    thresh = cv2.threshold(fgmask, T, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    (cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    for c in cnts:
      if cv2.contourArea(c) < args["min_area"] or cv2.contourArea(c) > args["max_area"]:
        continue

      (x, y, w, h) = cv2.boundingRect(c)
      cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Security Feed", frame)
    cv2.imshow("Mask", fgmask)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
      break

def MOG2(args, camera, out, T=10, blur=15):
  fgbg = cv2.createBackgroundSubtractorMOG2()

  while True:
    (grabbed, frame) = camera.read()

    if not grabbed:
      break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (blur, blur), 0)

    fgmask = fgbg.apply(gray)
    thresh = cv2.threshold(fgmask, T, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    (cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    for c in cnts:
      if cv2.contourArea(c) < args["min_area"] or cv2.contourArea(c) > args["max_area"]:
        continue

      (x, y, w, h) = cv2.boundingRect(c)
      cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Security Feed", frame)
    cv2.imshow("Mask", fgmask)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
      break

def GMG(args, camera, out, T=10, blur=15):
  fgbg = cv2.bgsegm.createBackgroundSubtractorGMG()

  while True:
    (grabbed, frame) = camera.read()

    if not grabbed:
      break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (blur, blur), 0)

    fgmask = fgbg.apply(gray)
    thresh = cv2.threshold(fgmask, T, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    (cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    for c in cnts:
      if cv2.contourArea(c) < args["min_area"] or cv2.contourArea(c) > args["max_area"]:
        continue

      (x, y, w, h) = cv2.boundingRect(c)
      cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Security Feed", frame)
    cv2.imshow("Mask", fgmask)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
      break

def KNN(args, camera, out, T=10, blur=15):
  fgbg = cv2.createBackgroundSubtractorKNN()

  while True:
    (grabbed, frame) = camera.read()

    if not grabbed:
      break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (blur, blur), 0)

    fgmask = fgbg.apply(gray)
    thresh = cv2.threshold(fgmask, T, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    (cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    for c in cnts:
      if cv2.contourArea(c) < args["min_area"] or cv2.contourArea(c) > args["max_area"]:
        continue

      (x, y, w, h) = cv2.boundingRect(c)
      cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Security Feed", frame)
    cv2.imshow("Mask", fgmask)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
      break

def LK_flow(args, camera, out, T=10, blur=15):
  feature_params = dict(maxCorners=100,
                        qualityLevel=0.3,
                        minDistance=3,
                        blockSize=7)
  lk_params = dict(winSize=(15, 15),
                  maxLevel=2,
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
  color = np.random.randint(0, 255, (100, 3))

  ret, frame1 = camera.read()
  prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
  p0 = cv2.goodFeaturesToTrack(prvs, mask=None, **feature_params)
  mask = np.zeros_like(frame1) 

  while True:
    (grabbed, frame) = camera.read()

    if not grabbed:
      break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # gray = cv2.GaussianBlur(gray, (blur, blur), 0)

    p1, st, err = cv2.calcOpticalFlowPyrLK(prvs, gray, p0, None, **lk_params)
    good_new = p1[st == 1]
    good_old = p0[st == 1]

    for i, (new, old) in enumerate(zip(good_new, good_old)):
      a, b = new.ravel()
      c, d = old.ravel()
      mask = cv2.line(mask, (int(a), int(b)), (int(c),int(d)), color[i].tolist(), 2)
      frame = cv2.circle(frame, (int(a),int(b)), 5, color[i].tolist(), -1)
    
    img = cv2.add(frame, mask)
    cv2.imshow('frame', img)

    prvs = gray
    p0 = good_new.reshape(-1, 1, 2)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
      break

def full_flow(args, camera, out, T=10, blur=15):
  ret, frame1 = camera.read()
  prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
  hsv = np.zeros_like(frame1)
  hsv[...,1] = 255

  while(1):
      ret, frame2 = camera.read()
      next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

      flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

      mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
      hsv[...,0] = ang*180/np.pi/2
      hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
      # bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
      
      cv2.imshow("Security Feed", frame2)
      cv2.imshow('frame2',hsv[...,2])
      k = cv2.waitKey(1) & 0xff
      if k == 27:
          break
      prvs = next
