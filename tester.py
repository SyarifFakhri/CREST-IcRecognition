import ICRecognition
import cv2
import imutils
import numpy as np

"""
The point of this code is to test the accuracy of the algorithm
take two folders, one called training, another one called testing
training is filled with images that we use as templates
testing is filled with images that we use to test
Then we get the accuracy based on how many it gets correct
We compare the response testing array to response training array
"""
numberOfTemplates = ICRecognition.numberOfTemplates
amountOfICs = ICRecognition.amountOfICs
widthImg = ICRecognition.widthImg
heightImg = ICRecognition.heightImg
threshToAddForGeneral = ICRecognition.threshToAddForGeneral
threshToAddForDetail = ICRecognition.threshToAddForDetail

responseTraining = []


#rn we assume response testing would be the same as response
responseTesting = []

totalCorrect = 0
total = 0

def getTesting():
    sampleNum = 1
    samples = []
    try:
        while True:
            imgTemplate = cv2.imread('templateCreatorTest' + str(sampleNum) + '.png')
            imgTemplate = cv2.resize(imgTemplate, (widthImg, heightImg), interpolation=cv2.INTER_LINEAR)

            # find the general area of the picture
            # threshVal, imgGrayAndBlurred, imgThresh = ICRecognition.binarizeImage(imgTemplate, threshToAddForGeneral)
            # imgInverted = cv2.bitwise_not(imgThresh)
            # # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            # # img = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel)
            #
            # imgContours, npaContours, npaHierarchy = cv2.findContours(imgInverted, cv2.RETR_EXTERNAL,
            #                                                           cv2.CHAIN_APPROX_SIMPLE)
            # contour = ICRecognition.returnLargestAreaOfContours(npaContours)
            #
            # roi = ICRecognition.deskewImageBasedOnContour(contour, imgGrayAndBlurred)
            #
            # # Then get the details
            # roi = cv2.resize(roi, (widthImg, heightImg), cv2.INTER_LINEAR)
            # ret, imgThresh = cv2.threshold(roi, threshVal + threshToAddForDetail, 255, cv2.THRESH_BINARY)

            # sample = extractFeatureFromImageForKNN(img)
            samples.append(imgTemplate)

            # if sampleNum % 5 == 0:
            # cv2.imshow("template" + str(sampleNum), imgTemplate)
            print("template:" + str(sampleNum) + " loaded")

            sampleNum = sampleNum + 1

    except:
        print("There are no more file templates! Last count was: ", sampleNum - 1)
        # cv2.imshow("originalTemplate", imgTemplate)

    # TODO - set the size based on automatic parameters
    return samples

#response testing would be filled with
for ICs in range(0, amountOfICs):
    for number in range(0, numberOfTemplates):
        responseTraining.append(ICs)

templates = ICRecognition.getTemplate()
testingPics = getTesting()

for count, pic in enumerate(testingPics):
    typeOfIC = ICRecognition.getICType(pic)
    responseTesting.append(typeOfIC)
    # cv2.imshow("TestTemplate" + str(count), pic)

for trainingResponse, testingResponse in zip(responseTraining, responseTesting):
    if trainingResponse == testingResponse:
        totalCorrect = totalCorrect + 1

    total = total + 1

print("The percentage of correct values is: ", totalCorrect/total * 100, "%")
print(responseTraining)
print(responseTesting)





