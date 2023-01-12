import cv2, numpy as np

img1 = cv2.imread('data/img.png')

kernel = np.array([[-1,-1,-1], [-1, 9, -1], [-1, -1, -1]])
res1 = cv2.filter2D(img1, -1, kernel)

ksize1 = 3; ksize2 = 15
img1_blur1 = cv2.blur(img1, (ksize1, ksize1))
img1_blur2 = cv2.blur(img1, (ksize2, ksize2))
res2 = cv2.subtract(img1.astype(np.uint16)*2, img1_blur1.astype(np.uint16))
res3 = cv2.subtract(img1.astype(np.uint16)*2, img1_blur2.astype(np.uint16))
res2 = res2.astype(np.uint8); res3 = res3.astype(np.uint8)

cv2.imshow('1', res1)
cv2.imshow('2', res2)
cv2.imshow('3', res3)

cv2.waitKey()
cv2.destroyAllWindows()