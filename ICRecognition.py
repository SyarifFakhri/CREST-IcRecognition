import cv2
from imutils import contours
import numpy as np
import imutils
from matplotlib import pyplot as plt
#import os
#import sys
#from PIL import Image


MIN_CONTOUR_AREA = 500
acceptedThreshold = 0.1
sigma = -50
#kernel = np.ones((1,1), np.uint8)
"COLOUR TEMPLATE"
cap = cv2.VideoCapture(0)

#TODO - Sort according to size first then sort according to type
#TODO - stop the conveyor belt when the ic is in view
#TODO - push the IC into the sorting boxes
#TODO -
template = {}
arrayOfResults = []

def binarizeImage(img):
    #This will extract features for template and second phase matching
    """This function converts the image to gray, puts a gaussian blur then does a thresh"""
    #Not sure why it's a thresh to zero, maybe a binarize would be better - but for now this is just refactoring so just roll with it
    #This should also be based on automatic parameter matching

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #Automatic thresholding - whatever the mean brightness is, just add a number
    # plt.hist(img.ravel(),256, [0,256]); plt.show()
    #
    # cv2.imshow("before histomgram equalized", img)
    # img = cv2.equalizeHist(img)

    mean = int(cv2.mean(img)[0])

    # cv2.imshow("histogram equalized", img)
    img = cv2.GaussianBlur(img, (5,5), 0)

    #note: if the image comes out too white, that means the threshold is too low, and if it comes out too black
    #it means the threshold is too high
    """
    thresholdValue = mean + sigma

    print("threshold value", thresholdValue)

    if thresholdValue < 10:
        #minimum is 10
        thresholdValue = 10"""

    # ret, img = cv2.threshold(img, thresholdValue,255, cv2.THRESH_BINARY)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 1)

    # img = cv2.bitwise_not(img)
    #
    return img

#TODO - Right now the template is only one, need to be able to add support for far more templates - need to return an array of images obv
def getTemplate():
    # Begin Getting of Template
    imgTemplate = cv2.imread('testTemplate0.png')

    # cv2.imshow("originalTemplate", imgTemplate)

    #TODO - set the size based on automatic parameters
    imgTemplate = imutils.resize(imgTemplate, width=300)

    img = binarizeImage(imgTemplate)

    cv2.imshow("template", img)

    return img

def deskewImageBasedOnContour(contour, img):
    """This function correcs the rotation of the IC"""
    # print("Largest Contour Area from npa contour", cv2.contourArea(contours))
    [x, y, w, h] = cv2.boundingRect(contour)
    # if h > w:
    # 	angle = angle + 90

    # cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    # cv2.drawContours(img,contours,-1, (0,255,0),1)
    # cv2.drawContours(img,npaContours,-1,(255,0,0),1)
    # need to deskew the contours
    # print(contours)

    rect = cv2.minAreaRect(contour)

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

    # cv2.rectangle(thresh, (x, y), (x + w, y + h), (255, 255, 0), 1)
    rows, cols, _ = img.shape
    # print(x,y,w,h)

    if x < 0:
        x = 0
    if x > rows:
        x = rows
    if y > cols:
        y = cols
    if y < 0:
        y = 0

    # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    # new center x is no longer the center here btw, it's now the x value of the rectangle
    rot = cv2.getRotationMatrix2D(center, angle, 1)
    # cv2.imshow("before rotation", img)

    img = cv2.warpAffine(img, rot, (cols, rows))
    # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
    # roiFinal = img[y:y + h, x:x + w]

    # set the ROI based on the rotated image
    img = img[y:y + h, x:x + w]
    # cv2.imshow("deskewed", img)
    return img

template[0] = getTemplate()

#End getting of template

while True:
    ret, img = cap.read()

    # img = cv2.imread('testTemplate2.png')
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))

    #Get the image threshold
    img = imutils.resize(img, width=300, height=600)
    cv2.imshow("Original", img)
    imgGrayTemplate = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)          # get grayscale image

    # something to play with here, where you use a closing operation followed by a dividing operation to get a uniform brightness
    # close = cv2.morphologyEx(imgGrayTemplate, cv2.MORPH_CLOSE, kernel)
    # cv2.imshow("closed", close)
    # div = np.float32(imgGrayTemplate)/(close)
    # cv2.imshow("divided",div)
    #TODO - implement automatic parameter handling for the threshold

    #First the algorithm searches for an IC as whole, before zooming into it
    imgBlurred = cv2.GaussianBlur(imgGrayTemplate, (5,5), 0)
    # imgThresh = cv2.adaptiveThreshold(imgBlurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    #
    # #open to remove noise
    # imgThresh = cv2.morphologyEx(imgThresh, cv2.MORPH_OPEN, kernel)
    # #close to find the largest contour
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(4,4))
    # imgThresh = cv2.morphologyEx(imgThresh, cv2.MORPH_CLOSE, kernel)
    ret, imgThresh = cv2.threshold(imgBlurred, 150,255, cv2.THRESH_BINARY)
    imgThresh = cv2.bitwise_not(imgThresh)
    cv2.imshow("thresh",imgThresh)

    imgThreshCopy = imgThresh.copy()

    imgContours, npaContours, npaHierarchy = cv2.findContours(imgThreshCopy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(img, npaContours, -1, (255,0,0),1)
    # cv2.imshow("contours",img)

    arrayOfContourAreas = []

    #This coding finds the largest Contour
    for npaContour in npaContours:
        (x, y, w, h) = cv2.boundingRect(npaContour)
        if cv2.contourArea(npaContour) > 100 and x > 0:
            arrayOfContourAreas.append(cv2.contourArea(npaContour))

    if arrayOfContourAreas != []:
        biggestIndex = np.argmax(arrayOfContourAreas)
        # print("contour found")
    else:
        #this is in case no contours are found, so that the array isn't empty
        arrayOfContourAreas.append(0)
        biggestIndex = 0
        # print("No contours were found")


    #makes sure that a contour exists, if not it'll crash
    #Once youve found the ROI you need to do new thresholding similar to how you found the template

    #This is the filtering process if it's an IC or not
    #TODO - Deal with multiple use cases
    #TODO 1 - Deal with multiple IC's on a single picture
    #TODO 2 - Deal with an IC being on the edge, before fully coming into frame
    #TODO 3 - Deal with varying areas of an IC and sort them before they even get into template matching

    #TODO 4 - this code is not efficient! Need to find a better way to get the largest contour area!
    if len(npaContours) != 0:
        #need to make sure the area is found if not it crashes
        found = False
        for npaContour in npaContours:
            if cv2.contourArea(npaContour) == arrayOfContourAreas[biggestIndex] and cv2.contourArea(npaContour) != 0:
                # print(cv2.contourArea(npaContour), arrayOfContourAreas[biggestIndex])
                contours = npaContour
                found = True
                # print("found!")
                #break the inner loop
                break
                #the point of this else and break is so that I can break out of 2 loops, the inner if loop and the outer for loop, if not break will only break the if loop and not the foor loop
            else:
                #continue if the inner loop wasn't broken
                continue
                #if not break the outer loop
            break
        # img = cv2.drawContours(img, contours, -1, (255, 0, 0), 2)
        # cv2.imshow("Largest area contour", img)

        if found:
            #deskew the image
            roi = deskewImageBasedOnContour(contours, img)

            # cv2.imshow("deskewed Image", roi)

            # imgGray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))
            #
            # #imgDenoised = cv2.fastNlMeansDenoising(roi, None,10,10,7,21)
            # imgblur = cv2.GaussianBlur(imgGray, (1,1), 0)
            # imgThresh = cv2.adaptiveThreshold(imgblur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
            #
            # #imgThresh = cv2.morphologyEx(imgThresh, cv2.MORPH_OPEN, kernel)
            imgThresh = binarizeImage(roi)

            roi = cv2.resize(imgThresh, (250,100))
            cv2.imshow('Image to match', roi)

            scores = []
            groupOutput = []

            for(templateCount, templateROI) in template.items():
                result = cv2.matchTemplate(roi, templateROI, cv2.TM_CCOEFF_NORMED)
                #print(result)
                (_,score,_,_) = cv2.minMaxLoc(result)
                print(score)
                scores.append(score)

            arrayOfResults.append(str(np.argmax(scores)))
            groupOutput.append(str(np.argmax(scores)))

            #return the most frequent of 10 results
            if len(arrayOfResults) == 10:
                (x, y, w, h) = cv2.boundingRect(contours)
                counts = np.bincount(arrayOfResults)
                string = str(np.argmax(counts))
                #only show if it's above the acceptable threshold
                if max(scores) > acceptedThreshold:
                    cv2.putText(img, "Type " + "".join(string) + "Area of contour: " + str(cv2.contourArea(contours)), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                else:
                    cv2.putText(img, "No IC found! " + "Area of contour: " + str(cv2.contourArea(contours)), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
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








