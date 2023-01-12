from matplotlib import pyplot as plt
import cv2, numpy as np

img1 = cv2.imread('data/img.png')

img1_HSV = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
img1_H, img1_S, img1_V = cv2.split(img1_HSV)

ch1 = [0]; ranges1 = [0, 256]; histSize1 = [256]; bin_x1 = np.arange(256)
mask1 = img1_H[110:150, 500:550]
hist_mask = cv2.calcHist([mask1], ch1, None, histSize1, ranges1)
bp = cv2.calcBackProject([img1_H], ch1, hist_mask, ranges1, 1)

print(img1_H.shape)
cv2.imshow('img1_H', img1_H)
cv2.imshow('img1_H2', img1_H[600:620, 200:240])
cv2.imshow('hist_norm', bp)
cv2.waitKey()
cv2.destroyAllWindows()