import cv2, numpy as np

img1 = cv2.imread('data/img.png', cv2.IMREAD_GRAYSCALE)

h, w = img1.shape
sort_pepper_ratio = (np.uint32)((h*w)*0.1)
sort_noise_x = np.full(shape=[sort_pepper_ratio], fill_value=0, dtype=np.uint16)
sort_noise_y = np.full(shape=[sort_pepper_ratio], fill_value=0, dtype=np.uint16)
pepper_noise_x = np.full(shape=[sort_pepper_ratio], fill_value=0, dtype=np.uint16)
pepper_noise_y = np.full(shape=[sort_pepper_ratio], fill_value=0, dtype=np.uint16)
gaussian_noise = np.full(shape=[h,w], fill_value=0, dtype=np.uint8)
cv2.randu(sort_noise_x, 0, w); cv2.randn(sort_noise_y, 0, h)
cv2.randu(pepper_noise_x, 0, w); cv2.randu(pepper_noise_y, 0, h)
cv2.randn(gaussian_noise, 0, 20)
sort_pepper_img = cv2.copyTo(img1, None)
gaussian_noise_img = cv2.add(img1, gaussian_noise)
for i in range(sort_pepper_ratio):
    sort_pepper_img[sort_noise_y[i], sort_noise_x[i]] = 255
    sort_pepper_img[pepper_noise_y[i], pepper_noise_x[i]] = 0

ksize1 = 3; ksize2 = 5
res1 = cv2.medianBlur(sort_pepper_img, ksize1)
res2 = cv2.medianBlur(sort_pepper_img, ksize2)
res3 = cv2.blur(gaussian_noise_img, (ksize1, ksize1))
res4 = cv2.GaussianBlur(gaussian_noise_img, (ksize1, ksize1), 0)
res5 = cv2.bilateralFilter(gaussian_noise_img, -1 , 20, ksize1)

cv2.imshow('1', res1)
cv2.imshow('2', res2)
cv2.imshow('3', res3)
cv2.imshow('4', res4)
cv2.imshow('5', res5)

cv2.waitKey()
cv2.destroyAllWindows()