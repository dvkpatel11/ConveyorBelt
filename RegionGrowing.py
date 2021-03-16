import numpy as np
import cv2


class regObj(object):
    def __init__(self, x, y):
        self.centerX = x
        self.centerY = y

    def getX(self):
        return self.centerX

    def getY(self):
        return self.centerY


def getGrayDiff(img, currentPoint, tmpPoint):
    return abs(int(img[currentPoint.centerX, currentPoint.centerY]) - int(img[tmpPoint.centerX, tmpPoint.centerY]))


def selectConnects(p):
    if p != 0:
        connects = [regObj(-1, -1), regObj(0, -1), regObj(1, -1), regObj(1, 0), regObj(1, 1), \
                    regObj(0, 1), regObj(-1, 1), regObj(-1, 0)]
    else:
        connects = [regObj(0, -1), regObj(1, 0), regObj(0, 1), regObj(-1, 0)]
    return connects


def regionGrow(img, seeds, thresh, p=1):
    height, weight = img.shape
    seedMark = np.zeros(img.shape)
    seedList = []
    for seed in seeds:
        seedList.append(seed)
    label = 1
    connects = selectConnects(p)
    while (len(seedList) > 0):
        currentPoint = seedList.pop(0)

        seedMark[currentPoint.centerX, currentPoint.centerY] = label
        for i in range(8):
            tmpX = currentPoint.centerX + connects[i].centerX
            tmpY = currentPoint.centerY + connects[i].centerY
            if tmpX < 0 or tmpY < 0 or tmpX >= height or tmpY >= weight:
                continue
            grayDiff = getGrayDiff(img, currentPoint, regObj(tmpX, tmpY))
            if grayDiff < thresh and seedMark[tmpX, tmpY] == 0:
                seedMark[tmpX, tmpY] = label
                seedList.append(regObj(tmpX, tmpY))
    return seedMark


img = cv2.imread('lean.png', 0)
seeds = [regObj(10, 10), regObj(82, 150), regObj(20, 200)]
binaryImg = regionGrow(img, seeds, 10)
cv2.imshow('',img)
cv2.imshow(' ', binaryImg)
cv2.waitKey(0)