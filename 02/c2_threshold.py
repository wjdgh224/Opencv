import numpy as np
import os
import cv2

img1 = cv2.imread('data/img.jpg', cv2.IMREAD_GRAYSCALE)

methods = [
            cv2.THRESH_BINARY,
            cv2.THRESH_BINARY_INV,
            cv2.THRESH_TRUNC,
            cv2.THRESH_TOZERO,
            cv2.THRESH_TOZERO_INV,
            cv2.THRESH_OTSU,
            cv2.THRESH_TRIANGLE,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C]
thres = 79; maxVal = 255
ress = []
for i in range(0, 7):
    ret, res = cv2.threshold(img1, thres, maxVal, methods[i])
    ress.append(res)
ress.append(cv2.adaptiveThreshold(img1, maxVal, methods[7], methods[0], 11, 0))
ress.append(cv2.adaptiveThreshold(img1, maxVal, methods[8], methods[0], 11, 0))
ress.append(cv2.adaptiveThreshold(img1, maxVal, methods[9], methods[0], 61, 0))
ress.append(cv2.adaptiveThreshold(img1, maxVal, methods[10], methods[0], 61, 0))
displays = [("input1", img1), ("res1", ress[0]), ("res2", ress[1]), ("res3", ress[2]), 
           ("res4", ress[3]), ("res5", ress[4]), ("res6", ress[5]), ("res7", ress[6]), 
           ("res8", ress[7]), ("res9", ress[8]), ("res10", ress[9]), ("res11", ress[10])]
for (name, out) in displays:
    cv2.imshow(name, out)

cv2.waitKey(0)
cv2.destroyAllWindows()
