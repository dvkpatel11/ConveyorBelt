#Import the library
import cv2
import numpy as np
import matplotlib.pyplot as plt

#Step
#1) Use Warp Perspective to Crop the Conveyor Belt Section
#2) Detect Black Objects and Contour 4 corner of the conveyor belt
corners = np.zeros((4,2),np.int)
counter = 0

#An empty function
def empty(var):
    pass

#Countour Area Track Bar
cv2.namedWindow("CountourSize")
cv2.resizeWindow("CountourSize",320,120)
cv2.createTrackbar("Min Area","CountourSize",500,300000,empty) #these values can be modified
minBlackPlasticCntSize = cv2.getTrackbarPos("Min Area","CountourSize");

#Register corners of the conveyor belt
def mousePoints(event,x,y,flags,params):
    global counter
    if event == cv2.EVENT_LBUTTONDOWN:
        # print(x,y)
        corners[counter] = x,y
        counter = counter + 1
        #print(corners)

cropImages = []

def getContours(imgCanny, imgContoured):
    global cropImages
    contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours)>0: #if there are contours detected
        for cnt in contours:
            cntArea = cv2.contourArea(cnt)
            if cntArea>=minBlackPlasticCntSize:
                hull = cv2.convexHull(cnt)  # implement convex hull
                cv2.drawContours(imgContoured, hull, -1, (255, 0, 0), 3)
                #approximate the bounding box
                cntPerimeter = cv2.arcLength(cnt,True)
                approx = cv2.approxPolyDP(cnt,0.02*cntPerimeter,True)
                x,y,w,h = cv2.boundingRect(approx)
                #bounding the detected plastic
                cv2.rectangle(imgContoured,(x,y),(x+w,y+h),(0,255,0),2)
                #Crop the images
                if len(cropImages)<5: #Because the number of cropped images is infinite
                    cropimg = imgContoured[y:(y+h),x:(x+w)]
                    cropImages.append(cropimg)

                #Center of the detected plastic
                cv2.circle(imgContoured,(x+(w//2),y+(h//2)),5,(0,255,0),cv2.FILLED)

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
        blackPlasticDetect = cv2.bitwise_and(belt,belt,mask=blackMask) #masked image
        #cv2.imshow("HSV", beltHSV)
        #cv2.imshow("Mask", blackMask)

        #Copy the belt to be contoured
        beltContoured = belt.copy()

        #Use Canny Edge Detection on the color thresholded belt

        beltDetectBlur = cv2.GaussianBlur(blackPlasticDetect,(7,7),1) #blurred image
        beltDetectGray = cv2.cvtColor(beltDetectBlur, cv2.COLOR_BGR2GRAY) #convert to grayscale
        ret, threshold = cv2.threshold(beltDetectBlur,10,255,cv2.THRESH_BINARY) #threshold grayscale image
        beltDetectCanny = cv2.Canny(threshold,50,50)
        kernel = np.ones((5, 5))
        beltDetectDilate = cv2.dilate(beltDetectCanny, kernel, iterations=1)
        getContours(beltDetectDilate,beltContoured)
        #Display cropped images
        for i in range(len(cropImages)):
            crop_w = int(cropImages[i].shape[1]*1)
            crop_h = int(cropImages[i].shape[0]*1)
            crop_dim = (crop_w,crop_h)
            cropResized = cv2.resize(cropImages[i],crop_dim,interpolation=cv2.INTER_AREA)
            cv2.imshow("Crop" + str(i),cropResized)
        cv2.imshow("Threshold Belt",threshold)
        # cv2.imshow("Blur",beltDetectBlur)
        # cv2.imshow("Canny",beltDetectCanny)

        #cv2.imshow("Black Plastics Detected", blackPlasticDetect)
        cv2.imshow("Black Plastics Contoured", beltContoured)


    #The frame of the webcam
    cv2.imshow("Frame",frame)


    #Press q on the keyboard to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


#Optional
# cap.release()
# cv2.destroyWindow()
