import cv2, numpy as np

img1 = cv2.imread('data/img.png')

h, w, _= img1.shape
c_h = h//2; c_w = w//2
rot_mat1 = cv2.getRotationMatrix2D((c_w, c_h), 45, 1)
rot_mat2 = cv2.getRotationMatrix2D((c_w, c_h), -45, 1)
res1 = cv2.warpAffine(img1, rot_mat1, (w,h))
res2 = cv2.warpAffine(img1, rot_mat2, (w,h))
res3 = cv2.rotate(img1, cv2.ROTATE_90_CLOCKWISE)
res4 = cv2.flip(img1, 1)
res5 = cv2.flip(img1, -1)

cv2.imshow('1', res1)
cv2.imshow('2', res2)
cv2.imshow('3', res3)
cv2.imshow('4', res4)
cv2.imshow('5', res5)

cv2.waitKey()
cv2.destroyAllWindows()