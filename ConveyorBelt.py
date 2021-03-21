import cv2
import numpy as np
import matplotlib.pyplot as plt
from centroidtracker import CentroidTracker #Class btained from pyImageSearch open source code
from skimage import data,filters,color,morphology
from skimage.segmentation import flood,flood_fill
from gpiozero import LED
from time import sleep

#Class to create detected objects and their centroids.
class regObj(object):
    def __init__(self, x, y):
        self.centerX = x
        self.centerY = y

    def getX(self):
        return self.centerX

    def getY(self):
        return self.centerY

corners = np.zeros((4,2),int)
counter = 0

#An empty function
def empty(var):
    pass

#Initialize centroid tracker and frame dimensions
ct = CentroidTracker()

#Countour Area Track Bar
cv2.namedWindow("CountourSize")
cv2.resizeWindow("CountourSize",320,120)
cv2.createTrackbar("Min Area","CountourSize",1500,30000,empty) #these values can be modified

minBlackPlasticCntSize = cv2.getTrackbarPos("Min Area","CountourSize");

#Register corners of the conveyor belt
def mousePoints(event,x,y,flags,params):
    global counter
    if event == cv2.EVENT_LBUTTONDOWN:
        # print(x,y)
        corners[counter] = x,y
        counter = counter + 1
        #print(corners)

def getContours(imgCanny, imgContoured):
    contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours)>0:
        for cnt in contours:
            if cv2.contourArea(cnt) >= minBlackPlasticCntSize:
                #get smallest bounding box
                rect1 = cv2.minAreaRect(cnt)
                box1 = cv2.boxPoints(rect1)
                box1 = np.int0(box1)
                cv2.drawContours(imgContoured, [box1], 0, (0, 2, 0), 2)
                #Get centroid
                M_individual = cv2.moments(cnt)
                centroid_x_individual = int(M_individual['m10'] / M_individual['m00'])
                centroid_y_individual = int(M_individual['m01'] / M_individual['m00'])
                cv2.drawContours(imgContoured,cnt,-1,(0,255,0),1)
                cntPerimeter = cv2.arcLength(cnt,True)
                approx = cv2.approxPolyDP(cnt,0.02*cntPerimeter,True)
                #Center of the detected plastic
                cv2.circle(imgContoured,(centroid_x_individual,centroid_y_individual),5,(0,255,0),cv2.FILLED)
                rects.append(regObj(centroid_x_individual, centroid_y_individual))
    return contours

def concatCont (rects, contours, imgContoured):
    large_contour_list = []
    if ((rects[0].centerX-rects[1].centerX)**2 + (rects[0].centerY-rects[1].centerY)**2)**0.5 < 1500:
        large_contour_list.append(contours[0])
        large_contour_list.append(contours[1])
        concat_cnts = np.concatenate(large_contour_list)
        rect2 = cv2.minAreaRect(concat_cnts)
        box2 = cv2.boxPoints(rect2)
        box2 = np.int0(box2)
        cv2.drawContours(imgContoured, [box2], 0, (255, 0, 0), 2)
        M = cv2.moments(concat_cnts)
        centroid_x = int(M['m10'] / M['m00'])
        centroid_y = int(M['m01'] / M['m00'])
        cv2.circle(imgContoured,(centroid_x,centroid_y),5,(255,0,0),cv2.FILLED)
        rects.pop(0)
        rects.pop(0)
        rects.insert(0,regObj(centroid_x,centroid_y))

#Check method to verify if an object center has reached the pump
def blowOff(rects, pumpPosY_min, pumpPosY_max):
    signalToPump = False
    if len(rects)>0:
        leadObjY = rects[0].centerY
        if (leadObjY < pumpPosY_max) and (leadObjY > pumpPosY_min):
            signalToPump = True
            rects.pop(0)
    return signalToPump

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

def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver

#Video Capture
cap = cv2.VideoCapture("capstone.mp4")
#cap = cv2.VideoCapture(0)
#Set the frame window dimensions
cap.set(3,640)
cap.set(4,480)
cap.set(10,100)

#Color Bars
cv2.namedWindow("ColorBars")
cv2.resizeWindow("ColorBars",640,240)
#Black has an HSV value of 0-255, 0-255, 0
cv2.createTrackbar("Hue Min","ColorBars",0,179,empty)
cv2.createTrackbar("Hue Max","ColorBars",179,179,empty)
cv2.createTrackbar("Sat Min","ColorBars",0,255,empty)
cv2.createTrackbar("Sat Max","ColorBars",255,255,empty)
cv2.createTrackbar("Val Min","ColorBars",0,255,empty)
cv2.createTrackbar("Val Max","ColorBars",60,255,empty) #Pick 90 as maximum value for black color
#Read the Video Capture
while (cap.isOpened()):
    success, frame = cap.read()
    if success == True:
        key = cv2.waitKey(3)
        # Click on the corners of the conveyor belt
        cv2.setMouseCallback("Frame", mousePoints)
        if key == ord('p'):
            cv2.waitKey(-1)
        if key == ord('q'):
            break
    else:
        break
    for i in range(0,4):
        cv2.circle(frame,(corners[i][0],corners[i][1]),3,(0,255,0),cv2.FILLED)
    #When 4 corners are selected
    if counter == 4:
        pts1 = np.float32([corners[0],corners[1],corners[2],corners[3]])
        #Make sure to follow the pattern
        beltwidth = np.float32(corners[2][0]-corners[0][0])
        belthheight = np.float32(corners[2][1]-corners[0][1])
        #Setting Pump position to be 0.75 *beltwidth
        pumpPosY_min = int(belthheight)*0.45
        pumpPosY_max = int(belthheight)*0.55
        pts2 = np.float32([[0,0],[beltwidth,0],[beltwidth,belthheight],[0,belthheight]])
        mtrx = cv2.getPerspectiveTransform(pts1,pts2)
        belt = cv2.warpPerspective(frame,mtrx,(beltwidth,belthheight))
        beltGray = cv2.cvtColor(belt,cv2.COLOR_BGR2GRAY)
        #Detect Black Objects on the Conveyor Belt
        beltHSV = cv2.cvtColor(belt,cv2.COLOR_BGR2HSV) #Convert the colorspace to HSV
        #Get the threshold from the colorbars window
        h_min = cv2.getTrackbarPos("Hue Min", "ColorBars")
        h_max = cv2.getTrackbarPos("Hue Max", "ColorBars")
        s_min = cv2.getTrackbarPos("Sat Min", "ColorBars")
        s_max = cv2.getTrackbarPos("Sat Max", "ColorBars")
        v_min = cv2.getTrackbarPos("Val Min", "ColorBars")
        v_max = cv2.getTrackbarPos("Val Max", "ColorBars")
        #Black Color Threshold
        lower_black = np.array([h_min,s_min,v_min])
        upper_black = np.array([h_max,s_max,v_max])
        blackMask = cv2.inRange(beltHSV,lower_black,upper_black)
        blackPlasticDetect = cv2.bitwise_and(belt,belt,mask=blackMask)
        #Copy the belt to be contoured
        beltContoured = belt.copy()
        #Mark line of pump position.
        cv2.line(beltContoured, (0, int(belthheight*0.45)), (beltwidth, int(belthheight*0.45)), (0, 0, 255), thickness=3)
        cv2.line(beltContoured, (0, int(belthheight * 0.55)), (beltwidth, int(belthheight * 0.55)), (0, 0, 255),thickness=3)
        imgCannyClose = cannyClose(blackPlasticDetect)
        rects = []
        #Get valid contours from canny detected image and display them
        contours = getContours(imgCannyClose,beltContoured)
        if len(rects)>1:
            concatCont(rects,contours, beltContoured)
        #Keep track of detected contours and store as objects
        objects = ct.update(rects)
        #Print object ID and centroid. Obtained from piImageSearch source codde.
        for (objectID, centroid) in objects.items():
            # draw both the ID of the object and the centroid of the
            # object on the output frame
            text = "ID {}".format(objectID)
            cv2.putText(belt, text, (centroid[0] - 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
        #sets a serial signal to True when a black object has reached the line of action of the pump
        #if blowOff(rects,pumpPosY_min,pumpPosY_max) == True:
            #print("true")

        imgStack = stackImages(1, ([belt, imgCannyClose, beltContoured,beltGray]))
        cv2.imshow("Process windows", imgStack)
        #cv2.imshow("Black Plastics Contoured", beltContoured)

    #The frame of the webcam
    cv2.imshow("Frame", frame)


    if key == ord('q'):
        break
    if key == ord('p'):
        cv2.waitKey(-1)  # wait until any key is pressed

#Optional
# cap.release()
# cv2.destroyWindow()
