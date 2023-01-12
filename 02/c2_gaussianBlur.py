import cv2, numpy as np

img1 = cv2.imread('data/img.png')

ksize1 = 7; ksize2 = 9
res1 = cv2.GaussianBlur(img1, (ksize1, ksize1), 0)
res2 = cv2.GaussianBlur(img1, (ksize2, ksize2), 0)
res3 = cv2.GaussianBlur(img1, (1, 21), 0)

cv2.imshow('1', res1)
cv2.imshow('2', res2)
cv2.imshow('3', res3)

cv2.waitKey()
cv2.destroyAllWindows()