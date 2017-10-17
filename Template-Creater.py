import cv2
from imutils import contours
import numpy as np
import imutils

MIN_CONTOUR_AREA = 500
#Begin Getting of Template
imgTemplate = cv2.imread('component.jpg')
imgTemplate = imutils.resize(imgTemplate, width=300)

imgGrayTemplate = cv2.cvtColor(imgTemplate, cv2.COLOR_BGR2GRAY)          # get grayscale image
imgBlurred = cv2.GaussianBlur(imgGrayTemplate, (5,5), 0)                        # blur

# filter image from grayscale to black and white
imgThresh = cv2.adaptiveThreshold(imgBlurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
cv2.imshow('thres',imgTemplate)

imgThreshCopy = imgThresh.copy()
imgContours, npaContours, npaHierarchy = cv2.findContours(imgThreshCopy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
digits = {}
template = {}

for (i,c) in enumerate(npaContours):
                                        # if contour is big enough to consider
    [x, y, w, h] = cv2.boundingRect(c)
    if cv2.contourArea(c) > MIN_CONTOUR_AREA:
        roi = imgTemplate[y:y+h, x:x+w]
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
    cv2.imwrite('template' + str(i) + '.png', template[i])



