import cv2, numpy as np

img1 = cv2.imread('data/img.jpg', cv2.IMREAD_GRAYSCALE)

v = np.full(shape=img1.shape, fill_value=100, dtype=np.uint8)
v_n = np.full(shape=img1.shape, fill_value=255, dtype=np.uint8) # img1과 크기가 같은 255값의 np

ress = []
ress.append(np.uint8(img1 + v))
ress.append(cv2.add(img1, v))
ress.append(cv2.subtract(v_n, img1))

cv2.imshow('a', ress[0])
cv2.imshow('b', ress[1])
cv2.imshow('c', ress[2])
cv2.waitKey()
cv2.destroyAllWindows()

# 영상의 명암비가 없어지는 문제