import cv2
from imutils import contours
import numpy as np
import imutils

cap = cv2.VideoCapture(1)

sigma = -50
MIN_CONTOUR_AREA = 500
thresholdValue = 60

def convertToHistogramFindMaxPeakAndReturnThresh(img):
    """calculates the histogram, smooths histogram, then plots the histogram of the image. Also plots the peaks"""
    """Purple is the threshold value, white is all the peaks,red is the two max peaks, green is the mean value"""
    img = cv2.GaussianBlur(img, (21,21), 0)
    histogram = cv2.calcHist([img], [0], None, [256], [0, 256])
    histogram = cv2.GaussianBlur(histogram, (21, 21), 0)
    #this implementation to find peaks is dam hacky, pls implement more elegant solution
    peaks = [0]*256

    histW, histH = 256, 500
    hist = np.zeros((histH, histW, 3), np.uint8)
    step = 1

    """find peaks"""
    for index in range(0, 256, step):
        try:
            if histogram[index] > histogram[index - step] and histogram[index] > histogram[index + step]:
                #ignore values that are too white, they're just background
                if index < 200:
                    peaks.insert(index, histogram[index][0])
                    cv2.line(hist, (index, 0), (index, histH), (255, 255, 255), 1)
        except:
            print ("histogram at index: ", index, "gave an error")

    # print(peaks)
    """find two maximum value peaks
    finalTwoPeaks = []
    for x in range(0,2):
        try:
            peak = np.argmax(peaks)
            peaks[np.argmax(peaks)] = 0
            finalTwoPeaks.append(peak)
        except:
            pass

    print(finalTwoPeaks)
    mean = int((finalTwoPeaks[0] + finalTwoPeaks[1])/2)
    """
    maxHist = np.argmax(peaks)
    thresh = 35
    #draw the entire histogram
    for index, value in enumerate(histogram):
        #normalize the values

        value = int((value/(max(histogram)))*500)

        cv2.line(hist, (index, histH), (index, histH - value), (255, 255, 0), 1)

    #draw the two maxes

    cv2.line(hist, (maxHist, 0), (maxHist, histH), (0, 0, 255), 1)
    cv2.line(hist, (maxHist + thresh, 0), (maxHist + thresh, histH), (0, 255, 0), 1)
    cv2.line(hist, (thresholdValue, 0), (thresholdValue, histH), (255, 0, 255), 1)
    cv2.imshow("histogram", hist)

    return maxHist + thresh


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

    # mean = int(cv2.mean(img)[0])

    mean = convertToHistogramFindMaxPeakAndReturnThresh(img)
    # cv2.imshow("histogram equalized", img)
    img = cv2.GaussianBlur(img, (5,5), 0)

    #note: if the image comes out too white, that means the threshold is too low, and if it comes out too black
    #it means the threshold is too high
    # thresholdValue = mean + sigma
    #
    # print("threshold value", thresholdValue)

    # if thresholdValue < 10:
    #     #minimum is 10
    #     thresholdValue = 10


    ret, imgStaticThresh = cv2.threshold(img, thresholdValue, 255, cv2.THRESH_BINARY)

    ret, img = cv2.threshold(img, mean, 255, cv2.THRESH_BINARY)
    cv2.imshow("staticThresh", imgStaticThresh)
    # img = cv2.bitwise_not(img)
    #
    return img

# capture first
while True:
    ret, img = cap.read()
    # img = cv2.imread("testTemplate8.png")

    img = imutils.resize(img, width=300, height=600)
    cv2.imshow("image", img)
    binarized = binarizeImage(img)
    cv2.imshow("binarized", binarized)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.imwrite("template.png", binarized)
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