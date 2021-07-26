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
    (cnts, _) = cv2.findContours(thresh.copy(),
                                 cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # print(np.array(cnts).shape)

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
    (cnts, _) = cv2.findContours(thresh.copy(),
                                 cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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
    (cnts, _) = cv2.findContours(thresh.copy(),
                                 cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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
    (cnts, _) = cv2.findContours(thresh.copy(),
                                 cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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
    (cnts, _) = cv2.findContours(thresh.copy(),
                                 cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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
    (cnts, _) = cv2.findContours(thresh.copy(),
                                 cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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
  prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
  p0 = cv2.goodFeaturesToTrack(prvs, mask=None, **feature_params)
  mask = np.zeros_like(frame1)

  print(p0)

  return

  while True:
    (grabbed, frame) = camera.read()

    if not grabbed:
      break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # gray = cv2.GaussianBlur(gray, (blur, blur), 0)

    p1, st, err = cv2.calcOpticalFlowPyrLK(
        prvs, gray, p0, None, **lk_params)
    good_new = p1[st == 1]
    good_old = p0[st == 1]

    for i, (new, old) in enumerate(zip(good_new, good_old)):
      a, b = new.ravel()
      c, d = old.ravel()
      mask = cv2.line(mask, (int(a), int(b)),
                      (int(c), int(d)), color[i].tolist(), 2)
      frame = cv2.circle(frame, (int(a), int(b)),
                         5, color[i].tolist(), -1)

    img = cv2.add(frame, mask)
    cv2.imshow('frame', img)

    prvs = gray
    p0 = good_new.reshape(-1, 1, 2)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
      break


def full_flow(args, camera, out, T=10, blur=15):
  ret, frame1 = camera.read()
  prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
  hsv = np.zeros_like(frame1)
  hsv[..., 1] = 255

  while(1):
    ret, frame2 = camera.read()
    next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(
        prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang*180/np.pi/2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    # bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

    cv2.imshow("Security Feed", frame2)
    cv2.imshow('frame2', hsv[..., 2])
    k = cv2.waitKey(1) & 0xff
    if k == 27:
      break
    prvs = next

class ViBe:
  '''
  ViBe运动检测，分割背景和前景运动图像
  '''

  def __init__(self, num_sam=20, min_match=2, radiu=20, rand_sam=16):
    self.defaultNbSamples = num_sam  # 每个像素的样本集数量，默认20个
    self.defaultReqMatches = min_match  # 前景像素匹配数量，如果超过此值，则认为是背景像素
    self.defaultRadius = radiu  # 匹配半径，即在该半径内则认为是匹配像素
    # 随机数因子，如果检测为背景，每个像素有1/defaultSubsamplingFactor几率更新样本集和领域样本集
    self.defaultSubsamplingFactor = rand_sam

    self.background = 0
    self.foreground = 255

  def __buildNeighborArray(self, img):
    '''
    构建一副图像中每个像素的邻域数组
    参数：输入灰度图像
    返回值：每个像素9邻域数组，保存到self.samples中
    '''
    height, width = img.shape
    self.samples = np.zeros(
        (self.defaultNbSamples, height, width), dtype=np.uint8)

    # 生成随机偏移数组，用于计算随机选择的邻域坐标
    ramoff_xy = np.random.randint(-1, 2, size=(2,
                                               self.defaultNbSamples, height, width))
    # ramoff_x=np.random.randint(-1,2,size=(self.defaultNbSamples,2,height,width))

    # xr_=np.zeros((height,width))
    xr_ = np.tile(np.arange(width), (height, 1))
    # yr_=np.zeros((height,width))
    yr_ = np.tile(np.arange(height), (width, 1)).T

    xyr_ = np.zeros((2, self.defaultNbSamples, height, width))
    for i in range(self.defaultNbSamples):
      xyr_[1, i] = xr_
      xyr_[0, i] = yr_

    xyr_ = xyr_+ramoff_xy

    xyr_[xyr_ < 0] = 0
    tpr_ = xyr_[1, :, :, -1]
    tpr_[tpr_ >= width] = width-1
    tpb_ = xyr_[0, :, -1, :]
    tpb_[tpb_ >= height] = height-1
    xyr_[0, :, -1, :] = tpb_
    xyr_[1, :, :, -1] = tpr_

    # xyr=np.transpose(xyr_,(2,3,1,0))
    xyr = xyr_.astype(int)
    self.samples = img[xyr[0, :, :, :], xyr[1, :, :, :]]

  def ProcessFirstFrame(self, img):
    '''
    处理视频的第一帧
    1、初始化每个像素的样本集矩阵
    2、初始化前景矩阵的mask
    3、初始化前景像素的检测次数矩阵
    参数：
    img: 传入的numpy图像素组，要求灰度图像
    返回值：
    每个像素的样本集numpy数组
    '''
    self.__buildNeighborArray(img)
    self.fgCount = np.zeros(img.shape)  # 每个像素被检测为前景的次数
    self.fgMask = np.zeros(img.shape)  # 保存前景像素

  def Update(self, img):
    '''
    处理每帧视频，更新运动前景，并更新样本集。该函数是本类的主函数
    输入：灰度图像
    '''
    height, width = img.shape
    # 计算当前像素值与样本库中值之差小于阀值范围RADIUS的个数，采用numpy的广播方法
    dist = np.abs((self.samples.astype(float) -
                   img.astype(float)).astype(int))
    dist[dist < self.defaultRadius] = 1
    dist[dist >= self.defaultRadius] = 0
    matches = np.sum(dist, axis=0)
    # 如果大于匹配数量阀值，则是背景，matches值False,否则为前景，值True
    matches = matches < self.defaultReqMatches
    self.fgMask[matches] = self.foreground
    self.fgMask[~matches] = self.background
    # 前景像素计数+1,背景像素的计数设置为0
    self.fgCount[matches] = self.fgCount[matches]+1
    self.fgCount[~matches] = 0
    # 如果某个像素连续50次被检测为前景，则认为一块静止区域被误判为运动，将其更新为背景点
    fakeFG = self.fgCount > 50
    matches[fakeFG] = False
    # 此处是该更新函数的关键
    # 更新背景像素的样本集，分两个步骤
    # 1、每个背景像素有1/self.defaultSubsamplingFactor几率更新自己的样本集
    # 更新样本集方式为随机选取该像素样本集中的一个元素，更新为当前像素的值
    # 2、每个背景像素有1/self.defaultSubsamplingFactor几率更新邻域的样本集
    # 更新邻域样本集方式为随机选取一个邻域点，并在该邻域点的样本集中随机选择一个更新为当前像素值
    # 更新自己样本集
    upfactor = np.random.randint(
        self.defaultSubsamplingFactor, size=img.shape)  # 生成每个像素的更新几率
    upfactor[matches] = 100  # 前景像素设置为100,其实可以是任何非零值，表示前景像素不需要更新样本集
    upSelfSamplesInd = np.where(upfactor == 0)  # 满足更新自己样本集像素的索引
    upSelfSamplesPosition = np.random.randint(
        self.defaultNbSamples, size=upSelfSamplesInd[0].shape)  # 生成随机更新自己样本集的的索引
    samInd = (upSelfSamplesPosition,
              upSelfSamplesInd[0], upSelfSamplesInd[1])
    # 更新自己样本集中的一个样本为本次图像中对应像素值
    self.samples[samInd] = img[upSelfSamplesInd]

    # 更新邻域样本集
    upfactor = np.random.randint(
        self.defaultSubsamplingFactor, size=img.shape)  # 生成每个像素的更新几率
    upfactor[matches] = 100  # 前景像素设置为100,其实可以是任何非零值，表示前景像素不需要更新样本集
    upNbSamplesInd = np.where(upfactor == 0)  # 满足更新邻域样本集背景像素的索引
    nbnums = upNbSamplesInd[0].shape[0]
    ramNbOffset = np.random.randint(-1, 2, size=(2, nbnums))  # 分别是X和Y坐标的偏移
    nbXY = np.stack(upNbSamplesInd)
    nbXY += ramNbOffset
    nbXY[nbXY < 0] = 0
    nbXY[0, nbXY[0, :] >= height] = height-1
    nbXY[1, nbXY[1, :] >= width] = width-1
    nbSPos = np.random.randint(self.defaultNbSamples, size=nbnums)
    nbSamInd = (nbSPos, nbXY[0], nbXY[1])
    self.samples[nbSamInd] = img[upNbSamplesInd]

  def getFGMask(self):
    '''
    返回前景mask
    '''
    return self.fgMask


def vibe(args, camera, out, T=10, blur=15):
  vibe = ViBe()
  (grabbed, frame) = camera.read()
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  # gray = cv2.GaussianBlur(gray, (blur, blur), 0)
  vibe.ProcessFirstFrame(gray)

  while True:
    (grabbed, frame) = camera.read()

    if not grabbed:
      break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # gray = cv2.GaussianBlur(gray, (blur, blur), 0)

    vibe.Update(gray)
    segMat = vibe.getFGMask()
    #　转为uint8类型
    segMat = segMat.astype(np.uint8)
    # thresh = cv2.threshold(fgmask, T, 255, cv2.THRESH_BINARY)[1]
    # thresh = cv2.dilate(thresh, None, iterations=2)
    # (cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    # for c in cnts:
    #   if cv2.contourArea(c) < args["min_area"] or cv2.contourArea(c) > args["max_area"]:
    #     continue

    #   (x, y, w, h) = cv2.boundingRect(c)
    #   cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Security Feed", frame)
    cv2.imshow("SegMat", segMat)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
      break

def diff_flow(args, camera, out, T=10, blur=15):
  feature_params = dict(maxCorners=100,
                      qualityLevel=0.3,
                      minDistance=3,
                      blockSize=7)
  lk_params = dict(winSize=(25, 25),
                   maxLevel=2,
                   criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
  color = np.random.randint(0, 255, (100, 3))

  lastFrame = None
  isDiff = True

  wait = 0

  while True:
    (grabbed, frame) = camera.read()

    if not grabbed:
      break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (blur, blur), 0)

    if lastFrame is None:
      lastFrame = gray
      mask = np.zeros_like(frame)
      continue

    if isDiff:
      delta = cv2.absdiff(lastFrame, gray)
      thresh = cv2.threshold(delta, T, 255, cv2.THRESH_BINARY)[1]
      thresh = cv2.dilate(thresh, None, iterations=2)
      (cnts, _) = cv2.findContours(thresh.copy(),
                                  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

      for c in cnts:
        if cv2.contourArea(c) < args["min_area"] or cv2.contourArea(c) > args["max_area"]:
          continue
        else:
          (x, y, w, h) = cv2.boundingRect(c)
          cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
          p0 = np.array(c, dtype=np.float32)
          p0 = np.mean(p0, axis=0)
          p0 = p0[np.newaxis, :]
          print(p0)
          isDiff = False
          # colorID = np.random.randint(0, 100)
          cv2.imshow('frame', frame)
          key = cv2.waitKey(100)
          
          # return
        # cv2.waitKey(0)
    else:
      p1, st, err = cv2.calcOpticalFlowPyrLK(
        lastFrame, gray, p0, None, **lk_params)
      # if len(p1):
        # print(1)
      good_new = p1[st == 1]
      good_old = p0[st == 1]

      for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (int(a), int(b)),
                        (int(c), int(d)), color[i].tolist(), 2)
        frame = cv2.circle(frame, (int(a), int(b)),
                          5, color[i].tolist(), -1)
      
      print(good_new)
      if good_new.shape[0] != 0 and wait < 6:
        new = good_new.reshape(-1, 1, 2)
        print()
        if np.sum((new - p0)**2) < 0.5:
          wait += 1
        p0 = good_new.reshape(-1, 1, 2)
      else:
        isDiff = True
        wait = 0
      # print(p0)

    frame = cv2.add(frame, mask)
    
    # p1, st, err = cv2.calcOpticalFlowPyrLK(
    #     lastFrame, gray, np.array(cnts), None, **lk_params)
    # good_new = p1[st == 1]
    # good_old = p0[st == 1]

    # for i, (new, old) in enumerate(zip(good_new, good_old)):
    #   a, b = new.ravel()
    #   c, d = old.ravel()
    #   mask = cv2.line(mask, (int(a), int(b)),
    #                   (int(c), int(d)), color[i].tolist(), 2)
    #   frame = cv2.circle(frame, (int(a), int(b)),
    #                      5, color[i].tolist(), -1)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
      break
    # img = cv2.add(frame, mask)
    out.write(frame)
    cv2.imshow('frame', frame)

    # p0 = good_new.reshape(-1, 1, 2)


    lastFrame = gray