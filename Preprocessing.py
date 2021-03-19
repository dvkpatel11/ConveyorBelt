import cv2
import numpy as np
import matplotlib.pyplot as plt

def cannyClose(img):
    bilateralfiltered = cv2.bilateralFilter(img, 3, 3, 3)
    gray = cv2.cvtColor(bilateralfiltered, cv2.COLOR_BGR2GRAY)
    v = np.median(gray)
    sigma = 0.33
    # ---- apply optimal Canny edge detection using the computed median----
    lower_thresh = int(max(0, (1.0 - sigma) * v))
    upper_thresh = int(min(255, (1.0 + sigma) * v))
    canny = cv2.Canny(gray, lower_thresh, upper_thresh)
    # Do a second layer of processing
    # beltDetectBlur = cv2.GaussianBlur(beltDetectCanny, (3, 3), 1)
    blur = cv2.bilateralFilter(canny, 3, 3, 3)
    # Remove small noise bits in image by dilating and eroding
    kernel = np.ones((5, 5))
    preprocessedimg = cv2.morphologyEx(blur, cv2.MORPH_CLOSE, kernel)
    return preprocessedimg