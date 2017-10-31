import cv2
import  numpy as np


ori = cv2.imread("testTemplate1.png")

img = cv2.cvtColor(ori, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray', img )
#
# blackHatKernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
# img2 = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, blackHatKernel)
# cv2.imshow("blackhat", img2)

#img = img - img2
# cv2.imshow("subtraction", img)



img = cv2.fastNlMeansDenoising(img, None, 5, 7, 21)

img = cv2.GaussianBlur(img, (7,7), 0)
cv2.imshow('blur', img )

ret, img = cv2.threshold(img, 100, 255, cv2.THRESH_TOZERO_INV)
cv2.imshow('thresh', img )

# img = cv2.bitwise_not(img)

img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21 ,2)
# img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 15 ,2)
cv2.imshow('adaptiveThresh', img )

# print(ret)


img = cv2.fastNlMeansDenoising(img, None, 5, 7, 21)
cv2.imshow("Denoise", img)

openingKernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4,5))
img = cv2.morphologyEx(img, cv2.MORPH_OPEN, openingKernel)
cv2.imshow("opening", img)

# dilationKernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))
# img = cv2.morphologyEx(img, cv2.MORPH_ERODE, dilationKernel)
# cv2.imshow("eroded", img)

ret, contours, __ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
newContours =[]
for contour in contours:
    if cv2.contourArea(contour) > 100 and cv2.contourArea(contour) < 1500:
        newContours.append(contour)
    # (x,y,w,h) = cv2.boundingRect(contour)
    # if

cv2.drawContours(ori, newContours, -1, (255,255,255), -1)
cv2.imshow("contours", ori)

# closingKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 7))
# img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, closingKernel)
# cv2.imshow("closing", img)

if cv2.waitKey(0) & 0xFF == ord('q'):
    cv2.destroyAllWindows()


