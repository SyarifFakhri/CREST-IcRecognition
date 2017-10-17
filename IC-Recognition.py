import cv2
from imutils import contours
import numpy as np
import imutils

#import os
#import sys
#from PIL import Image


MIN_CONTOUR_AREA = 500
#kernel = np.ones((1,1), np.uint8)
"COLOUR TEMPLATE"
cap = cv2.VideoCapture(0)

amountOfTemplatesPerIc = 2
amountOfIcsToDetect = 3
#right now the component images are the same, but this is more for proof of concept purposes
#The templates also needs to be named as 'component0.jpg', then 'component1.jpg' etc
#right now it's hard coded how many it detects, but we can make it not hardcoded later! - can use pathfinding for this
#The IC's to group together are the corresponding IC's of each component
#That means it would be IC 0 and IC 3, IC 1 and IC 4, IC 2 and IC 6 in the dictionary
#try take the mean values between them? or the highest value overall
#ideally each of the components should be in different images/folders - instead of 1 image, but this more a proof of concept
#TODO - Need to find a way to only initialize the components once! instead of it being in the for loop continously, which is not eficient!

template = {}
i = 0
#template should contain a 3*2 6 digit entry of what the compnonents are

for x in range(amountOfTemplatesPerIc):

    digits = {}
    #Begin Getting of Template
    string = 'component' + str(x) + '.jpg'
    imgTemplate = cv2.imread(string)
    imgTemplate = imutils.resize(imgTemplate, width=300)

    imgGrayTemplate = cv2.cvtColor(imgTemplate, cv2.COLOR_BGR2GRAY)          # get grayscale image
    imgBlurred = cv2.GaussianBlur(imgGrayTemplate, (5,5), 0)                        # blur

    # filter image from grayscale to black and white
    imgThresh = cv2.adaptiveThreshold(imgBlurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    cv2.imshow('thres',imgTemplate)

    imgThreshCopy = imgThresh.copy()
    imgContours, npaContours, npaHierarchy = cv2.findContours(imgThreshCopy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for (e,c) in enumerate(npaContours):
        # if contour is big enough to consider
        [x, y, w, h] = cv2.boundingRect(c)
        if cv2.contourArea(c) > MIN_CONTOUR_AREA:
            roi = imgTemplate[y:y+h, x:x+w]
            ###########
            refGray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            refBlur = cv2.GaussianBlur(refGray, (1,1), 0)
            refThresh = cv2.adaptiveThreshold(refBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
            ###########
            roi = cv2.resize(refThresh, (250, 100))
            digits[e] = roi

    #saving templates in sorted dictionary
    for c in digits:
        template[i] = digits[c]
        cv2.imshow('template' + str(i), template[i])
        i = i + 1

arrayOfResults = []

while True:
    #End getting of template

    ret, frame = cap.read()

    img = frame
    #img = cv2.imread('test0.jpg')
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))

    #Get the image threshold
    img = imutils.resize(img, width=300)
    imgGrayTemplate = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)          # get grayscale image

    # something to play with here, where you use a closing operation followed by a dividing operation to get a uniform brightness
    # close = cv2.morphologyEx(imgGrayTemplate, cv2.MORPH_CLOSE, kernel)
    # cv2.imshow("closed", close)
    # div = np.float32(imgGrayTemplate)/(close)
    # cv2.imshow("divided",div)

    imgBlurred = cv2.GaussianBlur(imgGrayTemplate, (5,5), 0)
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
            #cv2.drawContours(img,contours,-1, (0,255,0),1)
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
            cv2.imshow("roi2", roi)

            imgGray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))

            #imgDenoised = cv2.fastNlMeansDenoising(roi, None,10,10,7,21)
            imgblur = cv2.GaussianBlur(imgGray, (1,1), 0)
            imgThresh = cv2.adaptiveThreshold(imgblur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

            #imgThresh = cv2.morphologyEx(imgThresh, cv2.MORPH_OPEN, kernel)

            roi = cv2.resize(imgThresh, (250,100))
            cv2.imshow('roi', roi)

            scores = []
            groupOutput = []

            for(templateCount, templateRoi) in template.items():
                result = cv2.matchTemplate(roi, templateRoi, cv2.TM_CCOEFF_NORMED)
                #print(result)
                (_,score,_,_) = cv2.minMaxLoc(result)
                scores.append(score)
            #need to group together the results of the different templates of the ICs
            #rn this algorithm takes the mean of both results - should try max in both for e.g. and see the results for either!
            combinedScores = []
            for x in range(amountOfIcsToDetect):
                meanScore = (scores[x] + scores[x + amountOfIcsToDetect])/2
                combinedScores.append(meanScore)

            arrayOfResults.append(str(np.argmax(combinedScores)))
            groupOutput.append(str(np.argmax(combinedScores)))

            #return the most frequent of x results
            if len(arrayOfResults) == 1:
                counts = np.bincount(arrayOfResults)
                string = str(np.argmax(counts))
                cv2.putText(img, "Type " + "".join(string), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.imshow('final', img)
                arrayOfResults = []
            # cv2.putText(img, "Type " + "".join(groupOutput), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            # cv2.imshow('final', img)
            #print('type detected : ' + "".join(groupOutput))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
#out.release()
cv2.destroyAllWindows()

'''
cv2.waitKey(0)
cv2.destroyAllWindows()
'''








