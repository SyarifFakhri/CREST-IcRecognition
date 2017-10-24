import cv2
import  numpy as np


img = cv2.imread("testTemplate2.png")

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray', img )

img = cv2.fastNlMeansDenoising(img, None, 5, 7, 21)

img = cv2.GaussianBlur(img, (5,5), 0)
cv2.imshow('blur', img )

ret, img = cv2.threshold(img, 100, 255, cv2.THRESH_TOZERO_INV)
cv2.imshow('thresh', img )

img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11 ,2)
cv2.imshow('adaptiveThresh', img )

img = cv2.fastNlMeansDenoising(img, None, 5, 7, 21)
cv2.imshow("Denoise", img)

openingKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
img = cv2.morphologyEx(img, cv2.MORPH_OPEN, openingKernel)
cv2.imshow("opening", img)

closingKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 7))
img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, closingKernel)
cv2.imshow("closing", img)

if cv2.waitKey(0) & 0xFF == ord('q'):
    cv2.destroyAllWindows()
