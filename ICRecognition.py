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
threshToAddForDetail = 0
threshToAddForGeneral = 0
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
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)

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
        value = int((value/(max(histogram)))*500)
        cv2.line(hist, (index, histH), (index, histH - value), (255, 255, 0), 1)

    return hist

def convertToHistogramFindMaxPeakAndReturnThresh(img, threshToAdd):
    """calculates the histogram, smooths histogram, then plots the histogram of the image. Also plots the peaks"""
    """Purple is the threshold value, white is all the peaks,red is the two max peaks, green is the mean value"""
    img = cv2.GaussianBlur(img, (21,21), 0)
    histogram = cv2.calcHist([img], [0], None, [256], [0, 256])
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
                if index < 200:
                    peaks.insert(index, histogram[index][0])
                    cv2.line(hist, (index, 0), (index, histH), (255, 255, 255), 1)
        except:
            #print ("histogram at index: ", index, "gave an error")
            pass
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
    # cv2.imshow("histogram equalized", img)
    img = cv2.GaussianBlur(img, (5,5), 0)

    #use an automatic threshold, based on histogram peak value
    threshVal = convertToHistogramFindMaxPeakAndReturnThresh(img, withThreshToAdd)
    # ret, img = cv2.threshold(img, thresholdValue,255, cv2.THRESH_BINARY)
    ret, img = cv2.threshold(img, threshVal,255, cv2.THRESH_BINARY)

    return img

def getTemplate():
    global sampleNum
    # Begin Getting of Template
    #it will just cycle through images, possibly until it can't anymore
    samples = []
    try:
        while True:
            imgTemplate = cv2.imread('templateCreator' + str(sampleNum) + '.png')
            imgTemplate = imutils.resize(imgTemplate, width=300, height=200)

            #find the general area of the picture
            img = binarizeImage(imgTemplate, threshToAddForGeneral)
            img = cv2.bitwise_not(img)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            img = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel)

            imgContours, npaContours, npaHierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL,
                                                                      cv2.CHAIN_APPROX_SIMPLE)
            contour = returnLargestAreaOfContours(npaContours)

            img = deskewImageBasedOnContour(contour, imgTemplate)

            #Then get the details
            img = imutils.resize(img, width=300, height=200)
            img = binarizeImage(img, threshToAddForDetail)

            # sample = extractFeatureFromImageForKNN(img)
            samples.append(img)

            if sampleNum % 5 == 0:
                cv2.imshow("template" + str(sampleNum), img)
            print("template:" + str(sampleNum) + " loaded")
            sampleNum = sampleNum + 1

    except:
        print("There are no more file templates! Last count was: ", sampleNum - 1)

        # cv2.imshow("originalTemplate", imgTemplate)

    #TODO - set the size based on automatic parameters

    return samples

def deskewImageBasedOnContour(contour, img):
    """This function corrects the rotation of the IC"""
    """Make sure you give it a color image btw"""
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
        angle = angle - 90
        w, h = h, w

    x = int(cX) - int(w / 2)
    y = int(cY) - int(h / 2)

    # we only have the center so we need to get the x and the y

    # cv2.rectangle(thresh, (x, y), (x + w, y + h), (255, 255, 0), 1)
    rows, cols, __= img.shape
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

# def getMeanofAnArrayFromXtoY(array, x, y):
#     summation = 0
#     count = 0
#     #get the sum first
#     for index in range(x, y):
#         summation = summation + array[index]
#         count += 1
#     mean = (summation)/count
#
#     return mean
#

def returnLargestAreaOfContours(npaContours):
    """This function will take a thresholded image, find it's largest contour and return the ROI based on that"""

    # cv2.drawContours(img, npaContours, -1, (255,0,0),1)
    # cv2.imshow("contours",img)

    arrayOfContourAreas = []

    #This coding finds the largest Contour
    for npaContour in npaContours:
        (x, y, w, h) = cv2.boundingRect(npaContour)
        if cv2.contourArea(npaContour) > 100:
            arrayOfContourAreas.append(cv2.contourArea(npaContour))

    if arrayOfContourAreas != []:
        biggestIndex = np.argmax(arrayOfContourAreas)
        # print("contour found")

    #make sure that a contour exists, if not it'll crash
    #Once youve found the ROI you need to do new thresholding similar to how you found the template

    #This is the filtering process if it's an IC or not
    #TODO - Deal with multiple use cases
    #TODO 1 - Deal with multiple IC's on a single picture
    #TODO 2 - Deal with an IC being on the edge, before fully coming into frame
    #TODO 3 - Deal with varying areas of an IC and sort them before they even get into template matching

    #TODO 4 - this code is not efficient! Need to find a better way to get the largest contour area!
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

    return largestContour
        # img = cv2.drawContours(img, contours, -1, (255, 0, 0), 2)
        # cv2.imshow("Largest area contour", img)

# def extractFeatureFromImageForKNN(img):
#     """takes an image, resizes it into it's features and then returns an array"""
#     roismall = cv2.resize(img, (sampleX, sampleY))
#     sample = roismall.reshape((1, sampleX*sampleY))
#     return sample

# templateSamples = getTemplate()

# bigModel.train(samples, cv2.ml.ROW_SAMPLE, responses)
templates = getTemplate()

while True:

    response = []
    amountOfICs = 2
    numberOfTemplates = 5
    for ICs in range(0, amountOfICs):
        for number in range(0, numberOfTemplates):
            response.append(ICs)

    # cap.set(cv2.CAP_PROP_EXPOSURE, 1)
    ret, img = cap.read()

    # img = cv2.imread('testTemplate2.png')
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))

    #Get the image threshold
    img = imutils.resize(img, width=300, height=200)
    cv2.imshow("Original", img)
    imgThresh = binarizeImage(img, threshToAddForGeneral)         # get grayscale image
    imgThresh = cv2.bitwise_not(imgThresh)

    cv2.imshow("General thresh",imgThresh)

    imgThreshCopy = imgThresh.copy() #not sure why we make a copy here
    imgContours, npaContours, npaHierarchy = cv2.findContours(imgThreshCopy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largestContourInImage = returnLargestAreaOfContours(npaContours)
    cv2.drawContours(img, largestContourInImage, -1, (255,0,0), 2)

    if largestContourInImage is not None:
        #deskew the image
        roi = deskewImageBasedOnContour(largestContourInImage, img)
        # cv2.imshow("deskewed", roi)
        imgThresh = binarizeImage(roi, threshToAddForDetail)
        # cv2.imshow("imgThresh", imgThresh)
        #make sure that the roi is the same size as the templates
        #if the templates are bigger then the program will crash
        roi = imutils.resize(imgThresh, width=300)
        cv2.imshow('Image to match', roi)


        # roi = extractFeatureFromImageForKNN(roi)
        # roi = np.float32(roi)

        # retval, results, neighResp, dists = bigModel.findNearest(roi, k=k)
        # string = str(int((results[0][0])))

        # (x, y, w, h) = cv2.boundingRect(largestContourInImage)

        # cv2.putText(img, "Type " + "".join(string) + ": Area of contour: " + str(cv2.contourArea(largestContourInImage)),
        #             (x, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        #test
        # cv2.imshow("final", img)

        scores = []
        # groupOutput = []

        for(templateCount, templateROI) in enumerate(templates):
            result = cv2.matchTemplate(roi, templateROI, cv2.TM_CCOEFF_NORMED)
            #print(result)
            (_,score,_,_) = cv2.minMaxLoc(result)
            # print(score)
            scores.append(score)

        combinedResults = []
        # currentIndex = 0
        # #note: the response array here is an array that contains the number of responses each IC has. So [5,5] would mean
        # #a total of 10 ic templates and the first 5 correspond to IC 0, second 5 correspond to IC 1...etc
        # for x in response:
        #     mean = getMeanofAnArrayFromXtoY(scores, currentIndex, currentIndex + x)
        #     combinedResults.append(mean)
        #     currentIndex = currentIndex + x

        #use knn instead of just mean
        for x in range(0, k):
            maxIndex = np.argmax(scores)
            #put the closest inside the combined results array
            combinedResults.append(response.pop(maxIndex))

            #pop that one from the scores
            #what you should end up with is the n closest scores in the combinedresults array
            scores.pop(maxIndex)



        #need to combine the scores into one mean score per IC

        print(combinedResults)
        arrayOfResults.append(str(np.argmax(combinedResults)))
        # groupOutput.append(str(np.argmax(scores)))

        #return the most frequent of n results
        #this code will only run once every n frames
        if len(arrayOfResults) == 2:
            (x, y, w, h) = cv2.boundingRect(largestContourInImage)
            #np.bincount returns the most frequent result in the array of results
            counts = np.bincount(arrayOfResults)
            string = str(np.argmax(counts))
            #only show if it's above the acceptable threshold
            if max(scores) > acceptedThreshold:
                cv2.putText(img, "Type " + "".join(string) + "Area of contour: " + str(cv2.contourArea(largestContourInImage)), (x, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            else:
                cv2.putText(img, "No IC found! " + "Area of contour: " + str(cv2.contourArea(largestContourInImage)), (x, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            cv2.imshow('final', img)
            arrayOfResults = []
        # cv2.putText(img, "Type " + "".join(groupOutput), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        # cv2.imshow('final', img)
        #print('type detected : ' + "".join(groupOutput))

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








