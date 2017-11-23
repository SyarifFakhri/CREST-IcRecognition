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
#when you change it here change it in the template as well!

threshToAddForDetail = 30
threshToAddForGeneral = 20
histogramIgnoreValue = 150

amountOfICs = 2
numberOfTemplates = 10
widthImg = 300
heightImg = 200
k = 3

arrayOfResults = []

global sampleNum
sampleNum = 1

# samples = np.loadtxt('generalsamples.data',np.float32)
# responses = np.loadtxt('generalResponses.data', np.float32)

"""sampleX = 100
sampleY = 100

k = 3"""

"""#have one bigModel for large ICs and one small model for small ICs
bigModel = cv2.ml.KNearest_create()
"""
# try:
#cap.set(cv2.CAP_PROP_SETTINGS, 1)
# print("using camera from 1")

# except:
#     cap = cv2.VideoCapture(0)
#     ret, img = cap.read()
#     print("using camera from 0", ret)

# cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)

#TODO - Sort according to size first then sort according to type
#TODO - stop the conveyor belt when the ic is in view
#TODO - push the IC into the sorting boxes
#TODO - Deal with upside down cases!

# template = {}
# arrayOfResults = []

def drawHistogram(histogram, histW, histH):

    hist = np.zeros((histH, histW, 3), np.uint8)
    for index, value in enumerate(histogram):
        #normalize the values
        value = int((value/(max(histogram)))*histH)
        cv2.line(hist, (index, histH), (index, histH - value), (255, 255, 0), 1)

    return hist

def calcHistogram(img):
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    return hist

def convertToHistogramFindMaxPeakAndReturnThresh(img, threshToAdd):
    """calculates the histogram, smooths histogram, then plots the histogram of the image. Also plots the peaks"""
    """Purple is the threshold value, white is all the peaks,red is the two max peaks, green is the mean value"""
    img = cv2.GaussianBlur(img, (21,21), 0)
    histogram = calcHistogram(img)
    histogram = cv2.GaussianBlur(histogram, (21, 21), 0)
    #this implementation to find peaks is dam hacky, pls implement more elegant solution
    peaks = [0]*256

    step = 1
    histW, histH = 256, 500
    hist = drawHistogram(histogram, histW, histH)

    """find peaks"""
    for index in range(0, 256, step):
        try:
            if histogram[index] > histogram[index - step] and histogram[index] > histogram[index + step]:
                #ignore values that are too white, they're just background
                if index < histogramIgnoreValue:
                    peaks.insert(index, histogram[index][0])
                    cv2.line(hist, (index, 0), (index, histH), (255, 255, 255), 1)
        except:
            pass
            #print ("histogram at index: ", index, "gave an error")
    # print(peaks)
    maxHist = np.argmax(peaks)

    #draw the entire histogram
    #draw the two maxes

    cv2.line(hist, (maxHist, 0), (maxHist, histH), (0, 0, 255), 1)
    cv2.line(hist, (maxHist + threshToAdd, 0), (maxHist + threshToAdd, histH), (0, 255, 0), 1)
    cv2.imshow("histogram", hist)

    return maxHist + threshToAdd

def binarizeImage(img, withThreshToAdd):
    #This will extract features for template and second phase matching
    """This function converts the image to gray, puts a gaussian blur then does a thresh"""
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgGrayAndBlurred = cv2.GaussianBlur(img, (5,5), 0)

    #use an automatic threshold, based on histogram peak value
    threshVal = convertToHistogramFindMaxPeakAndReturnThresh(imgGrayAndBlurred, withThreshToAdd)

    ret, img = cv2.threshold(imgGrayAndBlurred, threshVal,255, cv2.THRESH_BINARY)

    return threshVal, imgGrayAndBlurred, img

def getTemplate():
    global sampleNum
    # Begin Getting of Template
    #it will just cycle through images, possibly until it can't anymore

    samples = []
    try:
        while True:
            imgTemplate = cv2.imread('templateCreator' + str(sampleNum) + '.png')
            # imgTemplate = cv2.resize(imgTemplate, (widthImg, heightImg), interpolation=cv2.INTER_LINEAR)
            #no need to resize cuz we save it as the correct size already

            #find the general area of the picture
            threshVal, imgGrayAndBlurred, imgThresh = binarizeImage(imgTemplate, threshToAddForGeneral)
            imgInverted = cv2.bitwise_not(imgThresh)
            # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            # img = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel)

            imgContours, npaContours, npaHierarchy = cv2.findContours(imgInverted, cv2.RETR_EXTERNAL,
                                                                      cv2.CHAIN_APPROX_SIMPLE)
            contour = returnLargestAreaOfContours(npaContours)
            if contour is not None:
                roi = deskewImageBasedOnContour(contour, imgGrayAndBlurred)

                #Then get the details
                roi = cv2.resize(roi,(widthImg, heightImg), cv2.INTER_LINEAR)
                ret, imgThresh = cv2.threshold(roi, threshVal + threshToAddForDetail, 255, cv2.THRESH_BINARY)
                # cv2.imshow("Roi", imgThresh)

                # sample = extractFeatureFromImageForKNN(img)
                samples.append(imgThresh)

                # if sampleNum % 5 == 0:
                    # cv2.imshow("template" + str(sampleNum), imgTemplate)
                print("template:" + str(sampleNum) + " loaded")
            else:
                print("Template: ", str(sampleNum), " - contour is none")

            sampleNum = sampleNum + 1

    except:
        print("There are no more file templates! Last count was: ", sampleNum - 1)
        # cv2.imshow("originalTemplate", imgTemplate)

    #TODO - set the size based on automatic parameters
    return samples

def deskewImageBasedOnContour(contour, img):
    """This function corrects the rotation of the IC"""
    """Make sure you give it a color image btw"""
    # need to deskew the contours
    rect = cv2.minAreaRect(contour)

    center = rect[0]
    angle = rect[2]

    # get the roi of the rotated bounding box - this is also the exact dimensions of the straightened box
    cX, cY = center
    w, h = rect[1]

    w = int(w)
    h = int(h)

    if w < h:
        angle = angle - 90
        w, h = h, w

    x = int(cX) - int(w / 2)
    y = int(cY) - int(h / 2)

    # we only have the center so we need to get the x and the y

    # cv2.rectangle(thresh, (x, y), (x + w, y + h), (255, 255, 0), 1)
    rows, cols = img.shape
    # print(x,y,w,h)

    if x < 0:
        x = 0
    if x > rows:
        x = rows
    if y > cols:
        y = cols
    if y < 0:
        y = 0

    # new center x is no longer the center here btw, it's now the x value of the rectangle
    rot = cv2.getRotationMatrix2D(center, angle, 1)

    img = cv2.warpAffine(img, rot, (cols, rows))

    # set the ROI based on the rotated image
    img = img[y:y + h, x:x + w]
    # print("angle: ", angle)
    # print("x: ", x, "y: ", y, "w: ", w, "h: ",h)

    return img

def returnLargestAreaOfContours(npaContours):
    """This function will take a thresholded image, find it's largest contour and return the ROI based on that"""

    # cv2.drawContours(img, npaContours, -1, (255,0,0),1)
    # cv2.imshow("contours",img)

    arrayOfContourAreas = []

    #This coding finds the largest Contour
    for npaContour in npaContours:
        if cv2.contourArea(npaContour) > MIN_CONTOUR_AREA:
            arrayOfContourAreas.append(cv2.contourArea(npaContour))

    if arrayOfContourAreas != []:
        biggestIndex = np.argmax(arrayOfContourAreas)

    #make sure that a contour exists, if not it'll crash
    #Once youve found the ROI you need to do new thresholding similar to how you found the template

    #This is the filtering process if it's an IC or not

    largestContour = None

    try:
        #need to make sure the area is found that's why we use except
        for npaContour in npaContours:
            if cv2.contourArea(npaContour) == arrayOfContourAreas[biggestIndex] and cv2.contourArea(npaContour) != 0:
                # print(cv2.contourArea(npaContour), arrayOfContourAreas[biggestIndex])
                largestContour = npaContour
                # print("found!")
                #break the inner loop
                break
                #the point of this else and break is so that I can break out of 2 loops, the inner if loop and the outer for loop, if not break will only break the if loop and not the foor loop
            else:
                #continue if the inner loop wasn't broken
                continue
                #if not break the outer loop
            break
    except:
        return None
    # print("The largest contour area is: ", cv2.contourArea(largestContour))
    return largestContour

def getICType(img, templates):
    # this is creating the response array
    response = []
    for ICs in range(0, amountOfICs):
        for number in range(0, numberOfTemplates):
            response.append(ICs)

    # Get the image threshold
    imgResized = cv2.resize(img, (widthImg, heightImg), interpolation=cv2.INTER_LINEAR)
    cv2.imshow("resized", imgResized)

    # TODO - DO THE GRAY, HISTOGRAM VALUE, THRESHOLD CALCULATIONS, BLUR CALCULATIONS ONCE ONLY THEN PASS IT AROUND
    threshVal, imgGrayAndBlurred, imgThresh = binarizeImage(imgResized, threshToAddForGeneral)  # get grayscale image
    adaptiveThresh = cv2.adaptiveThreshold(imgGrayAndBlurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11,2)
    ret, naiveThreshold = cv2.threshold(imgGrayAndBlurred, 100,255, cv2.THRESH_BINARY)
    imgInverted = cv2.bitwise_not(imgThresh)

    cv2.imshow("General thresh",imgInverted)
    # cv2.imshow("adaptive thersh", adaptiveThresh)
    # cv2.imshow("Naive threshold", naiveThreshold)
    imgContours, npaContours, npaHierarchy = cv2.findContours(imgInverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largestContourInImage = returnLargestAreaOfContours(npaContours)

    # (x, y, w, h) = cv2.boundingRect(largestContourInImage)
    # cv2.rectangle(imgGrayAndBlurred, (x, y), (x + w, y + h), (255, 255, 255), 2)

    # cv2.drawContours(imgResized, largestContourInImage, -1, (255,0,0), 2)

    # cv2.imshow("contours", imgResized)

    if largestContourInImage is not None:
        # print("The area of the contour is: ", cv2.contourArea(largestContourInImage))
        # deskew the image
        roi = deskewImageBasedOnContour(largestContourInImage, imgGrayAndBlurred)
        cv2.imshow("deskewed", roi)

        # cv2.drawContours(imgGrayAndBlurred, largestContourInImage, -1, (255,255,255), 1)

        # cv2.imshow("grayAndBlurred", imgGrayAndBlurred)

        # cv2.imshow("deskewed", roi)
        ret, imgThresh = cv2.threshold(roi, threshVal + threshToAddForDetail, 255, cv2.THRESH_BINARY)
        cv2.imshow("imgThresh", imgThresh)

        # make sure that the roi is the same size as the templates
        # if the templates are bigger then the program will crash
        roi = cv2.resize(imgThresh, (widthImg, heightImg), interpolation=cv2.INTER_LINEAR)
        # cv2.imshow('Image to match', roi)
        cv2.imshow("roi", roi)
        scores = []

        for (templateCount, templateROI) in enumerate(templates):
            result = cv2.matchTemplate(roi, templateROI, cv2.TM_CCOEFF_NORMED)
            # print(result)
            (_, score, _, _) = cv2.minMaxLoc(result)
            # print(score)
            scores.append(score)

        maxScores = max(scores)
        combinedResults = []

        # use knn instead of just mean
        for x in range(0, k):
            maxIndex = np.argmax(scores)
            # put the closest inside the combined results array
            combinedResults.append(response.pop(maxIndex))
            # pop that one from the scores
            # what you should end up with is the n closest scores in the combinedresults array
            scores.pop(maxIndex)

        # need to combine the scores into one mean score per IC

        print(combinedResults)

        count = np.bincount(combinedResults)
        maximumCount = np.argmax(count)
        arrayOfResults.append(maximumCount)
        # groupOutput.append(str(np.argmax(scores)))

        # return the most frequent of n results
        # this code will only run once every n frames
        if len(arrayOfResults) == 1:
            # (x, y, w, h) = cv2.boundingRect(largestContourInImage)
            # np.bincount returns the most frequent result in the array of results
            counts = np.bincount(arrayOfResults)
            maximumCount = np.argmax(counts)
            # string = str(maximumCount)
            # only show if it's above the acceptable threshold
            if maxScores > acceptedThreshold:
                # cv2.putText(img, "Type " + "".join(string) + "Area of contour: " + str(cv2.contourArea(largestContourInImage)), (x, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                print("The IC type is: ", maximumCount)
                return maximumCount
            else:
                # cv2.putText(img, "No IC found! " + "Area of contour: " + str(cv2.contourArea(largestContourInImage)), (x, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                print("No IC found!")
                return None
                # cv2.imshow('final', img)
                # arrayOfResults = []
                # cv2.putText(img, "Type " + "".join(groupOutput), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                # cv2.imshow('final', img)
                # print('type detected : ' + "".join(groupOutput))

    else:
        print("No contour found!")


if __name__ == '__main__':

    templates = getTemplate()

    cap = cv2.VideoCapture(1)

    while True:
        ret, img = cap.read()
        getICType(img, templates)

        # if cv2.waitKey(1) & 0xFF == ord('q'):
        if cv2.waitKey(1) == 27: #escape to break
            break

    cap.release()
    #out.release()
    cv2.destroyAllWindows()


'''
cv2.waitKey(0)
cv2.destroyAllWindows()
'''








