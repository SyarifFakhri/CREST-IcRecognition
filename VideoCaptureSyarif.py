import cv2
from imutils import contours
import numpy as np
import imutils
#import os
#import sys
#from PIL import Image


MIN_CONTOUR_AREA = 500
#kernel = np.ones((1,1), np.uint8)

cap = cv2.VideoCapture(0)
#TODO - Need to find a way to only initialize the components once! instead of it being in the for loop continously, which is not eficient!

#Begin Getting of Template
img = cv2.imread('component.jpg')
img = imutils.resize(img, width=300)

imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)          # get grayscale image
imgBlurred = cv2.GaussianBlur(imgGray, (5,5), 0)                        # blur

# filter image from grayscale to black and white
imgThresh = cv2.adaptiveThreshold(imgBlurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

cv2.imshow('thres',img)
imgThreshCopy = imgThresh.copy()
imgContours, npaContours, npaHierarchy = cv2.findContours(imgThreshCopy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
digits = {}
template = {}

for (i,c) in enumerate(npaContours):
                                        # if contour is big enough to consider
    [x, y, w, h] = cv2.boundingRect(c)
    if cv2.contourArea(c) > MIN_CONTOUR_AREA:
        roi = img[y:y+h, x:x+w]
        ###########
        refgray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        refblur = cv2.GaussianBlur(refgray, (1,1), 0)
        refthresh = cv2.adaptiveThreshold(refblur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        ###########
        roi = cv2.resize(refthresh, (250, 100))
        digits[i] = roi

#saving templates in sorted dictionary
for (i,c) in enumerate(digits):
    template[i] = digits[c]
    cv2.imshow('template' + str(i), template[i])
#getting image test and zoom the get the contour

while True:
    #End getting of template

    ret, frame = cap.read()

    img = frame
    #img = cv2.imread('test2skewed.jpg')
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))

    #Get the image threshold
    img = imutils.resize(img, width=300)
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)          # get grayscale image

    # something to play with here, where you use a closing operation followed by a dividing operation to get a uniform brightness
    # close = cv2.morphologyEx(imgGray, cv2.MORPH_CLOSE, kernel)
    # cv2.imshow("closed", close)
    # div = np.float32(imgGray)/(close)
    # cv2.imshow("divided",div)

    imgBlurred = cv2.GaussianBlur(imgGray, (5,5), 0)
    imgThresh = cv2.adaptiveThreshold(imgBlurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    #open to remove noise
    imgThresh = cv2.morphologyEx(imgThresh, cv2.MORPH_OPEN, kernel)
    #close to find the largest contour
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(4,4))
    imgThresh = cv2.morphologyEx(imgThresh, cv2.MORPH_CLOSE, kernel)


    cv2.imshow("thresh",imgThresh)

    imgThreshCopy = imgThresh.copy()
    imgContours, npaContours, npaHierarchy = cv2.findContours(imgThreshCopy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #cv2.drawContours(img, npaContours, -1, (255,0,0),1)
    cv2.imshow("contours",img)
    arrayOfContourAreas = []

    #This coding finds the largest Contour
    for npaContour in npaContours:
        if cv2.contourArea(npaContour) > 100:
            arrayOfContourAreas.append(cv2.contourArea(npaContour))

    if arrayOfContourAreas != []:
        biggestIndex = np.argmax(arrayOfContourAreas)
        #print("contour found")
    else:
        #this is in case no contours are found, so that the array isn't empty
        arrayOfContourAreas.append(0)
        biggestIndex = 0
        #print("No contours were found")
    #TODO - this code is not efficient! Need to find a better way to get the largest contour area!

    #makes sure that a contour exists, if not it'll crash

    if len(npaContours) != 0:
        #need to make sure the area is found if not it crashes
        found = False
        for npaContour in npaContours:
            if cv2.contourArea(npaContour) == arrayOfContourAreas[biggestIndex] and cv2.contourArea(npaContour) != 0:
                print(cv2.contourArea(npaContour), arrayOfContourAreas[biggestIndex])
                contours = npaContour
                found = True
                print("found!")
                #break the inner loop
                break
                #the point of this else and break is so that I can break out of 2 loops, the inner if loop and the outer for loop, if not break will only break the if loop and not the foor loop
            else:
                #continue if the inner loop wasn't broken
                continue
                #if not break the outer loop
            break



        if found == True:
            #print("Largest Contour Area from npa contour", cv2.contourArea(contours))
            [x,y,w,h] = cv2.boundingRect(contours)
            # if h > w:
            # 	angle = angle + 90

            #cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            cv2.drawContours(img,contours,-1, (0,255,0),1)
            #cv2.drawContours(img,npaContours,-1,(255,0,0),1)
            #need to deskew the contours
            #print(contours)

            rect = cv2.minAreaRect(contours)

            center = rect[0]
            angle = rect[2]

            # get the roi of the rotated bounding box - this is also the exact dimensions of the straightened box
            cX, cY = center
            w, h = rect[1]

            w = int(w)
            h = int(h)

            if w < h:
                angle = angle + 90
                w, h = h, w

            x = int(cX) - int(w / 2)
            y = int(cY) - int(h / 2)

            # we only have the center so we need to get the x and the y

            #cv2.rectangle(thresh, (x, y), (x + w, y + h), (255, 255, 0), 1)
            rows, cols = imgThresh.shape
            #print(x,y,w,h)

            if x < 0:
                x = 0
            if x > rows:
                x = rows
            if y > cols:
                y = cols
            if y < 0:
                y = 0

            #cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # new center x is no longer the center here btw, it's now the x value of the rectangle
            rot = cv2.getRotationMatrix2D(center, angle, 1)
            cv2.imshow("before rotation",img)

            #TODO - set it so that only the ROI rotates not everything!
            img = cv2.warpAffine(img, rot, (cols, rows))
            #cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
            # roiFinal = img[y:y + h, x:x + w]

            #set the ROI based on the rotated image
            roi = img[y:y+h, x:x+w]
            #cv2.imshow("roi2", roi)

            imggray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            #imgblur = cv2.GaussianBlur(imggray, (1,1), 0)

            imgthresh = cv2.adaptiveThreshold(imggray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

            roi = cv2.resize(imgthresh, (250,100))
            cv2.imshow('roi', roi)

            scores = []
            groupOutput = []

            for(templateCount, templateROI) in template.items():
                result = cv2.matchTemplate(roi, templateROI, cv2.TM_CCOEFF)
                #print(result)
                (_,score,_,_) = cv2.minMaxLoc(result)
                #print(score)
                scores.append(score)

            groupOutput.append(str(np.argmax(scores)))
            #print('type detected : ' + "".join(groupOutput))
            cv2.putText(img, "Type " + "".join(groupOutput), (x, y ), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.imshow('final', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
#out.release()
cv2.destroyAllWindows()
'''
cv2.waitKey(0)
cv2.destroyAllWindows()
'''








