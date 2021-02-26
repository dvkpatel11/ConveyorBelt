import cv2
import numpy as np
import matplotlib.pyplot as plt

cap = cv2.VideoCapture(0)
fgbg = cv2.createBackgroundSubtractorMOG2()


def getContours(imgCanny, imgContoured):
    global  minContourArea
    #Get the list of contours from canny edged image
    contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #Check if any contour detected
    if len(contours)>0:
        for cnt in contours:
            #Contour Area used to filter noise
            cntArea = cv2.contourArea(cnt)
            if cntArea>minContourArea:
                cv2.drawContours(imgContoured,cnt,-1,(255,0,0),2)
                #Approximate bounding box
                cntPerimeter = cv2.arcLength(cnt,True)
                approx = cv2.approxPolyDP(cnt,0.02*cntPerimeter,True)
                x,y,w,h = cv2.boundingRect(approx)
                #bounding the detected plastic
                cv2.rectangle(imgContoured,(x,y),(x+w,y+h),(0,255,0),3)
                #Center of the detected plastic
                cv2.circle(imgContoured,(x+(w//2),y+(h//2)),5,(0,255,0),cv2.FILLED)

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


while True:
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)

    ##Get contours
    crop = []
    contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #Check if any contour detected
    if len(contours)>0:
        for cnt in contours:
            #Contour Area used to filter noise
            cntArea = cv2.contourArea(cnt)
            if cntArea>1000:
                cv2.drawContours(frame,cnt,-1,(255,0,0),2)
                #Approximate bounding box
                cntPerimeter = cv2.arcLength(cnt,True)
                approx = cv2.approxPolyDP(cnt,0.02*cntPerimeter,True)
                x,y,w,h = cv2.boundingRect(approx)
                #bounding the detected plastic
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
                #Center of the detected plastic
                cv2.circle(frame,(x+(w//2),y+(h//2)),5,(0,255,0),cv2.FILLED)
                #crop contours
                crop.append(frame[y:y+h,x:x+w])

        for i in range(len(crop)):
            cv2.imshow("Crop"+str(i),crop[i])

    cv2.imshow("Frame",frame)
    cv2.imshow("BGFG",fgmask)
    # Press q on the keyboard to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# cap.release()
# cv2.destroyAllWindows()