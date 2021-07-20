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