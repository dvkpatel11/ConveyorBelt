#Import the library
import cv2
import numpy as np
import matplotlib.pyplot as plt

#4 corners of the conveyor belt
corners = np.zeros((4,2),np.int)
counter = 0

#Register corners of the conveyor belt
def mousePoints(event,x,y,flags,params):
    global counter
    if event == cv2.EVENT_LBUTTONDOWN:
        # print(x,y)
        corners[counter] = x,y
        counter = counter + 1
        print(corners)

#Video Capture
cap = cv2.VideoCapture(0)

#Read the Video Capture
while True:
    _, frame = cap.read()

    #The frame of the webcam
    cv2.imshow("Frame",frame)

    #Click on the corners of the conveyor belt
    cv2.setMouseCallback("Frame",mousePoints)
    #When 4 corners are selected
    if counter == 4:
        w,h = 250,350 #Can be modified for the setting
        pts1 = np.float32([corners[0],corners[1],corners[2],corners[3]])
        #Make sure to follow the pattern
        pts2 = np.float32([[0,0],[w,0],[w,h],[0,h]])
        mtrx = cv2.getPerspectiveTransform(pts1,pts2)
        belt = cv2.warpPerspective(frame,mtrx,(w,h))
        cv2.imshow("Cropped Belt",belt)

    #Press q on the keyboard to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


#Optional
# cap.release()
# cv2.destroyWindow()


