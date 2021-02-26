#Import the library
import cv2
import numpy as np
import matplotlib.pyplot as plt
from centroidtracker import CentroidTracker #Class btained from pyImageSearch open source code

#Class to create detected objects and their centroids.
class regObj:
    def __init__(self, paramCenterX, paramCenterY):
        self.centerX = paramCenterX
        self.centerY = paramCenterY

#Step
#1) Use Warp Perspective to Crop the Conveyor Belt Section
#2) Detect Black Objects and Contour 4 corner of the conveyor belt
corners = np.zeros((4,2),np.int)
counter = 0

#An empty function
def empty(var):
    pass

#Initialize centroid tracker and frame dimensions
ct = CentroidTracker()

#Countour Area Track Bar
cv2.namedWindow("CountourSize")
cv2.resizeWindow("CountourSize",320,120)
cv2.createTrackbar("Min Area","CountourSize",500,30000,empty) #these values can be modified

minBlackPlasticCntSize = cv2.getTrackbarPos("Min Area","CountourSize");

#Register corners of the conveyor belt
def mousePoints(event,x,y,flags,params):
    global counter
    if event == cv2.EVENT_LBUTTONDOWN:
        # print(x,y)
        corners[counter] = x,y
        counter = counter + 1
        #print(corners)

#def secDetect(imgCanny):
#    contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #for cnt in contours:
    #    cntArea = cv2.contourArea(cnt)
    #    if cntArea>=minBlackPlasticCntSize:
    #        cntPerimeter = cv2.arcLength(cnt,True)
    #        approx = cv2.approxPolyDP(cnt,0.02*cntPerimeter,True)
    #        x,y,w,h = cv2.boundingRect(approx)
    #        crop_img = imgCanny[y:y + h, x:x + w]
#    kernel = np.ones((5, 5))
#    cv2.morphologyEx(imgCanny, cv2.MORPH_OPEN, kernel)
#    cv2.morphologyEx(imgCanny, cv2.MORPH_CLOSE, kernel)

def getContours(imgCanny, imgContoured):
    contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #cv2.drawContours(imgContoured,contours,-1,(0,255,0),2) ;Already showing contours in the for loop
    #maxContour = max(contours,key=cv2.contourArea)
    # x, y, w, h = cv2.boundingRect(maxContour)
    # cv2.rectangle(imgContoured, (x, y), (x + w, y + h), (0, 255, 0), 3)
    # cv2.circle(imgContoured, (x + (w // 2), y + (h // 2)), 5, (0, 255, 0), cv2.FILLED)
    for cnt in contours:
        cntArea = cv2.contourArea(cnt)
        if cntArea>=minBlackPlasticCntSize:
            cv2.drawContours(imgContoured,cnt,-1,(0,255,0),2)
            cntPerimeter = cv2.arcLength(cnt,True)
            approx = cv2.approxPolyDP(cnt,0.02*cntPerimeter,True)
            x,y,w,h = cv2.boundingRect(approx)
            #bounding the detected plastic
            cv2.rectangle(imgContoured,(x,y),(x+w,y+h),(0,255,0),3)
            #Center of the detected plastic
            cv2.circle(imgContoured,(x+(w//2),y+(h//2)),5,(0,255,0),cv2.FILLED)
            #Register a valid contour obj
            rects.append(regObj((x+(w//2)),(y+(h//2))))

#Check method to verify if an object center has reached the pump
def blowOff(paramPosX, pumpPosX):
    signalToPump = False
    if (paramPosX == pumpPosX):
        signalToPump = True
    return signalToPump

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
cap = cv2.VideoCapture(0)
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
cv2.createTrackbar("Val Max","ColorBars",90,255,empty) #Pick 90 as maximum value for black color

#Read the Video Capture
while True:
    success, frame = cap.read()

    #Click on the corners of the conveyor belt
    cv2.setMouseCallback("Frame",mousePoints)

    for i in range(0,4):
        cv2.circle(frame,(corners[i][0],corners[i][1]),3,(0,255,0),cv2.FILLED)

    #When 4 corners are selected
    if counter == 4:
        pts1 = np.float32([corners[0],corners[1],corners[2],corners[3]])
        #Make sure to follow the pattern
        # beltwidth = 250
        # belthheight = 350
        beltwidth = np.float32(corners[2][0]-corners[0][0])
        belthheight = np.float32(corners[2][1]-corners[0][1])
        #Setting Pump position to be 0.75 *beltwidth
        pumpPosX = int(beltwidth)*0.75
        pts2 = np.float32([[0,0],[beltwidth,0],[beltwidth,belthheight],[0,belthheight]])
        mtrx = cv2.getPerspectiveTransform(pts1,pts2)
        belt = cv2.warpPerspective(frame,mtrx,(beltwidth,belthheight))
        #cv2.imshow("Cropped Belt",belt)
        #Detect Black Objects on the Conveyor Belt
        beltHSV = cv2.cvtColor(belt,cv2.COLOR_BGR2HSV) #Convert the colorspace to HSV
        #Get the threshold from the colorbars window
        h_min = cv2.getTrackbarPos("Hue Min", "ColorBars")
        h_max = cv2.getTrackbarPos("Hue Max", "ColorBars")
        s_min = cv2.getTrackbarPos("Sat Min", "ColorBars")
        s_max = cv2.getTrackbarPos("Sat Max", "ColorBars")
        v_min = cv2.getTrackbarPos("Val Min", "ColorBars")
        v_max = cv2.getTrackbarPos("Val Max", "ColorBars")
        # print(h_min,h_max,s_min,s_max,v_min,v_max)
        #Black Color Threshold
        lower = np.array([h_min,s_min,v_min])
        upper = np.array([h_max,s_max,v_max])
        blackMask = cv2.inRange(beltHSV,lower,upper)
        blackPlasticDetect = cv2.bitwise_and(belt,belt,mask=blackMask)
        #Copy the belt to be contoured
        beltContoured = belt.copy()
        #Mark line of pump position.
        cv2.line(beltContoured, (int(beltwidth*0.75),0), (int(beltwidth*0.75),belthheight), (0, 0, 255), thickness=3)
        #Use Canny Edge Detection on the color thresholded belt
        beltBilateralFiltered = cv2.bilateralFilter(blackPlasticDetect, 3, 3, 3)
        beltDetectGray = cv2.cvtColor(beltBilateralFiltered, cv2.COLOR_BGR2GRAY)
        v = np.median(beltDetectGray)
        sigma = 0.33
        # ---- apply optimal Canny edge detection using the computed median----
        lower_thresh = int(max(0, (1.0 - sigma) * v))
        upper_thresh = int(min(255, (1.0 + sigma) * v))
        beltDetectCanny = cv2.Canny(beltDetectGray,lower_thresh,upper_thresh)
        #Do a second layer of processing
        #beltDetectBlur = cv2.GaussianBlur(beltDetectCanny, (3, 3), 1)
        beltDetectBlur = cv2.bilateralFilter(beltDetectCanny, 3, 3, 3)
        # Remove small noise bits in image by dilating and eroding
        kernel = np.ones((5, 5))
        imgCannyClose = cv2.morphologyEx(beltDetectBlur, cv2.MORPH_CLOSE, kernel)
        #List of detected bounding rectangles
        rects = []
        #Get valid contours from canny detected image and display them
        getContours(imgCannyClose,beltContoured)
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
        #signalToPump = blowOff(leadObjPosX, pumpPosX)
        imgStack = stackImages(1, ([belt, beltDetectCanny, imgCannyClose, beltContoured]))
        cv2.imshow("Process windows", imgStack)
        #cv2.imshow("Black Plastics Contoured", beltContoured)

    #The frame of the webcam
    cv2.imshow("Frame", frame)

    #Press q on the keyboard to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#Optional
# cap.release()
# cv2.destroyWindow()
