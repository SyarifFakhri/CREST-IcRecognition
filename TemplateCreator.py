import cv2
from imutils import contours
import numpy as np
import imutils

cap = cv2.VideoCapture(1)
#150 images roughly translates to about 40 mb
#
# cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
# cap.set(cv2.CAP_PROP_BRIGHTNESS, 0)  # Doesn't work lel

MIN_CONTOUR_AREA = 500
# thresholdValue = 60
#when you change it here change it in the recognizer as well!
threshToAddForDetail = 35
threshToAddForGeneral = 20

# sampleX = 100
# sampleY = 100
# samples = np.empty((0, sampleX * sampleY))

#this is actually KNN responses array, it should actually correspond to the type inside the template samples
# responses = [[0]*10,[1]*10]

#convert to numpy to make it faster
# responses = np.array(responses, np.float32)
# responses = responses.reshape((responses.size,1))
#End getting of template

count = 1

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
    histW, histH = 256, 200
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

# capture first
while True:
    ret, imgTemplate = cap.read()

    # img = cv2.imread("testTemplate8.png")
    imgTemplate = imutils.resize(imgTemplate, width=300)

    # find the general area of the picture
    img = binarizeImage(imgTemplate, threshToAddForGeneral)
    img = cv2.bitwise_not(img)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    img = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel)
    cv2.imshow("general binarization", img)

    try:
        imgContours, npaContours, npaHierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL,
                                                                  cv2.CHAIN_APPROX_SIMPLE)
        contour = returnLargestAreaOfContours(npaContours)

        img = deskewImageBasedOnContour(contour, imgTemplate)

        # Then get the details
        img = imutils.resize(img, width=300)
        img = binarizeImage(img, threshToAddForDetail)
        cv2.imshow("image", imgTemplate)

        cv2.imshow("binarized", img)

    except:
        print("error occurred")
    key = cv2.waitKey(1)

    if key == 32:
        #space to save the image
        # cv2.imwrite('templateCreator' + str(count) +'.png', imgTemplate)
        # sample = extractFeatureFromImageForKNN(img)
        cv2.imwrite("templateCreator" + str(count) + ".png", imgTemplate)
        print("Image saved" + str(count))
        count += 1

    if key == 27:  # (escape to quit)
        cv2.destroyAllWindows()
        break

#then do the output stuff


"""#Begin Getting of Template
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



"""