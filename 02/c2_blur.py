import cv2, numpy as np

img1 = cv2.imread('data/img.png')

ksize1 = 3; ksize2 = 5; ksize3 = 7; ksize4 = 9
kernel = np.full(shape=[ksize4,ksize4], fill_value=1, dtype=np.float32) / (ksize4*ksize4)
res1 = cv2.blur(img1, (ksize1, ksize1))
res2 = cv2.blur(img1, (ksize2, ksize2))
res3 = cv2.boxFilter(img1, -1, (ksize3, ksize3))
res4 = cv2.filter2D(img1, -1, kernel)
res5 = cv2.boxFilter(img1, -1, (1, 21))


cv2.imshow('1', res1)
cv2.imshow('2', res2)
cv2.imshow('3', res3)
cv2.imshow('4', res4)
cv2.imshow('5', res5)

cv2.waitKey()
cv2.destroyAllWindows()