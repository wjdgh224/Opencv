import cv2, numpy as np
import matplotlib.pyplot as plt
import math

image = cv2.imread('data/img.jpg').astype(np.float32)/255


image = np.full((480, 640, 3), 255, np.uint8) #  세로, 가로, 레벨(BGR) : 왼쪽에서 쌓인 종이를 본다.
cv2.imshow('white', image)
cv2.waitKey()
cv2.destroyAllWindows()

image = np.full((480, 640, 3), (0, 0, 255), np.uint8)
cv2.imshow('red', image)
cv2.waitKey()
cv2.destroyAllWindows()

image.fill(0) # 기존 행렬 값 동일하게 할당
cv2.imshow('black', image)
cv2.waitKey()
cv2.destroyAllWindows()

image[240, 160] = image[240, 320] = image[240, 280] = (255, 255, 255)
cv2.imshow('black with white pixels', image)
cv2.waitKey()
cv2.destroyAllWindows()

image[:,:,0] = 255
cv2.imshow('blue with white pixels', image)
cv2.waitKey()
cv2.destroyAllWindows()

image[:,320,:] = 255
cv2.imshow('blue with white line', image)
cv2.waitKey()
cv2.destroyAllWindows()

image[100:600, 100:200, 2] = 255
cv2.imshow('image', image)
cv2.waitKey()
cv2.destroyAllWindows()


print('Shape:', image.shape)
print('Data type:', image.dtype)

cv2.imshow('image', image)
cv2.waitKey()
cv2.destroyAllWindows()

image = image.astype(np.float32) / 255
print('Shape:', image.shape)
print('Data type:', image.dtype)

cv2.imshow('image', np.clip(image*2, 0, 1)) # 0~1조정
cv2.waitKey()
cv2.destroyAllWindows()

image = (image * 255).astype(np.uint8)
print('Shape:', image.shape)
print('Data type:', image.dtype)

cv2.imshow('image', image)
cv2.waitKey()
cv2.destroyAllWindows()


# 데이터 영속성, 저장
mat = np.random.rand(100, 100).astype(np.float32)
print('Shape:', mat.shape)
print('Data type:', mat.dtype)

np.savetxt('mat.csv', mat)

mat = np.loadtxt('mat.csv').astype(np.float32)
print('Shape:', mat.shape)
print('Data type:', mat.dtype)


print('Shape:', image.shape)

image[:,:,[0,2]] = image[:,:,[2,0]]
cv2.imshow('blue_and_red_swapped', image)
cv2.waitKey()
cv2.destroyAllWindows()

image[:,:,[0,2]] = image[:,:,[2,0]]
image[:,:,0] = (image[:,:,0]*0.9).clip(0,1)
image[:,:,1] = (image[:,:,1]*1.1).clip(0,1)
cv2.imshow('image', image)
cv2.waitKey()
cv2.destroyAllWindows()

print('Shape:', image.shape)
print('Data type:', image.dtype)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print('Converted to grayscale')
print('Shape:', gray.shape)
print('Data type', gray.dtype)
cv2.imshow('gray', gray)
cv2.waitKey()
cv2.destroyAllWindows()

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
print('Converted to HSV')
print('Shape:', hsv.shape)
print('Data type', hsv.dtype)
cv2.imshow('hsv', hsv)
cv2.waitKey()
cv2.destroyAllWindows()

hsv[:,:,2] *= 2
from_hsv = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
print('Converted back to BGR from HSV')
print('Shape:', from_hsv.shape)
print('Data type', from_hsv.dtype)
cv2.imshow('from_hsv', from_hsv)
cv2.waitKey()
cv2.destroyAllWindows()


# 감마 보정 : 이미지 픽셀 강도
gamma = 0.5
corrected_image = np.power(image, gamma)

cv2.imshow('image', image)
cv2.imshow('corrected_image', corrected_image)
cv2.waitKey()
cv2.destroyAllWindows()


# 평균 분산
image -= image.mean()
image /= image.std()


grey = cv2.imread('data/img.jpg', 0)
cv2.imshow('original grey', grey)
cv2.waitKey()
cv2.destroyAllWindows()

hist, bins = np.histogram(grey, 256, [0, 255])

plt.fill(hist)
plt.xlabel('pixel value')
plt.show()


# 평활화
grey = cv2.imread('data/img.jpg', 0)
cv2.imshow('original grey', grey)
cv2.waitKey()
cv2.destroyAllWindows()

grey_eq = cv2.equalizeHist(grey)

hist, bins=  np.histogram(grey_eq, 256, [0, 255])
plt.fill_between(range(256), hist, 0)
plt.xlabel('pixel value')
plt.show()

cv2.imshow('equalized grey', grey_eq)
cv2.waitKey()
cv2.destroyAllWindows()

color = cv2.imread('data/img.jpg')
hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)

hsv[..., 2] = cv2.equalizeHist(hsv[..., 2])
color_eq = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
cv2.imshow('original color', color)

cv2.imshow('equalized color', color_eq)
cv2.waitKey()
cv2.destroyAllWindows()


# 노이즈 제거
noised = (image + 0.2 * np.random.rand(*image.shape).astype(np.float32))
noised = noised.clip(0, 1)
plt.imshow(noised[:,:,[2,1,0]])
plt.show()

gauss_blur = cv2.GaussianBlur(noised, (7,7), 0)
plt.imshow(gauss_blur[:,:,[2,1,0]])
plt.show()

median_blur = cv2.medianBlur((noised*255).astype(np.uint8), 7)
plt.imshow(gauss_blur[:,:,[2,1,0]])
plt.show()

bilat = cv2.bilateralFilter(noised, -1, 0.3, 10)
plt.imshow(median_blur[:,:,[2,1,0]])
plt.show()


# 이미지 경사도 : 미분 -> 급격한 변화를 찾아 에지를 찾는다.

image = cv2.imread('data/img.jpg', 0)

dx = cv2.Sobel(image, cv2.CV_32F, 1, 0)
dy = cv2.Sobel(image, cv2.CV_32F, 0, 1)

plt.figure(figsize=(8, 3))
plt.subplot(131)
plt.axis('off')
plt.title('image')
plt.imshow(image, cmap='gray')
plt.subplot(132)
plt.axis('off')
plt.imshow(dx, cmap='gray')
plt.title(r'$\frac{dI}{dx}$') # 분수 표현
plt.subplot(133)
plt.axis('off')
plt.title(r'$\frac{dI}{dx}$')
plt.imshow(dy, cmap='gray')
plt.tight_layout()
plt.show()

# 자체 필터 : 높은 주파수 강조
image = cv2.imread('data/img.jpg')

KSIZE = 11
ALPHA = 2

kernel = cv2.getGaussianKernel(KSIZE, 0)
kernel = -ALPHA * kernel @ kernel.T
kernel[KSIZE//2, KSIZE//2] += 1 + ALPHA

filtered = cv2.filter2D(image, -1, kernel)

plt.figure(figsize=(8,4))
plt.subplot(121)
plt.axis('off')
plt.title('image')
plt.imshow(image[:, :, [2,1,0]])
plt.subplot(122)
plt.axis('off')
plt.title('filltered')
plt.imshow(filtered[:,:,[2,1,0]])
plt.tight_layout()
plt.show()


# 가버 필터 커널(이미지 에지 검출 용이)
image = cv2.imread('data/img.jpg', 0).astype(np.float32)/255

kernel = cv2.getGaborKernel((21,21), 5, 1, 10, 1, 0, cv2.CV_32F)
kernel /= math.sqrt((kernel*kernel).sum())

filtered = cv2.filter2D(image, -1, kernel)

plt.figure(figsize=(8, 3))
plt.subplot(131)
plt.axis('off')
plt.title('image')
plt.imshow(image, cmap='gray')
plt.subplot(132)
plt.title('kernel')
plt.imshow(kernel, cmap='gray')
plt.subplot(133)
plt.axis('off')
plt.title('filtered')
plt.imshow(filtered, cmap='gray')
plt.tight_layout()
plt.show()


# 푸리에를 이용한 공간 도메인과 주파수 도메인 간 변환
image = cv2.imread('data/img.jpg', 0).astype(np.float32)/255

fft = cv2.dft(image, flags=cv2.DFT_COMPLEX_OUTPUT)

shifted = np.fft.fftshift(fft, axes=[0, 1])
magnitude = cv2.magnitude(shifted[:,:,0], shifted[:,:,1])
magnitude = np.log(magnitude)

plt.axis('off')
plt.imshow(magnitude, cmap='gray')
plt.tight_layout()
plt.show()

restored = cv2.idft(fft, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)


# 주파수 조작
image = cv2.imread('data/img.jpg', 0).astype(np.float32)/255

fft = cv2.dft(image, flags=cv2.DFT_COMPLEX_OUTPUT)

fft_shift = np.fft.fftshift(fft, axes=[0, 1])

sz = 25
mask = np.zeros(fft_shift.shape, np.uint8)
mask[mask.shape[0]//2-sz:mask.shape[0]//2+sz,
mask.shape[1]//2-sz:mask.shape[1]//2+sz, :] = 1
fft_shift *= mask

fft = np.fft.ifftshift(fft_shift, axes=[0, 1])

filtered = cv2.idft(fft, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)

plt.figure()
plt.subplot(121)
plt.axis('off')
plt.title('original')
plt.imshow(image, cmap='gray')
plt.subplot(122)
plt.axis('off')
plt.title('no high frequencies')
plt.imshow(filtered, cmap='gray')
plt.tight_layout()
plt.show()


# 임계 처리
image = cv2.imread('data/img.jpg', 0)

thr, mask = cv2.threshold(image, 200, 1, cv2.THRESH_BINARY)
print('Threshold used:', thr)

adapt_mask = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 10)

plt.figure(figsize=(10, 3))
plt.subplot(131)
plt.axis('off')
plt.title('original')
plt.imshow(image, cmap='gray')
plt.subplot(132)
plt.axis('off')
plt.title('binary threshold')
plt.imshow(mask, cmap='gray')
plt.subplot(133)
plt.axis('off')
plt.title('adaptive threshold')
plt.imshow(adapt_mask, cmap='gray')
plt.tight_layout()
plt.show()


# 형태 연산 : 침식, 팽창
image = cv2.imread('data/img.jpg', 0)
_, binary = cv2.threshold(image, -1, 1, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

eroded = cv2.morphologyEx(binary, cv2.MORPH_ERODE, (3,3), iterations=10)
dilated = cv2.morphologyEx(binary, cv2.MORPH_DILATE, (3,3), iterations=10)

opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)), iterations=5)
closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)), iterations=5)

grad = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

plt.figure(figsize=(10, 10))
plt.subplot(231)
plt.axis('off')
plt.title('binary')
plt.imshow(binary, cmap='gray')
plt.subplot(232)
plt.axis('off')
plt.title('erode 10 times')
plt.imshow(eroded, cmap='gray')
plt.subplot(233)
plt.axis('off')
plt.title('dilate 10 times')
plt.imshow(dilated, cmap='gray')
plt.subplot(234)
plt.axis('off')
plt.title('open 5 times')
plt.imshow(opened, cmap='gray')
plt.subplot(235)
plt.axis('off')
plt.title('close 5 times')
plt.imshow(closed, cmap='gray')
plt.subplot(236)
plt.axis('off')
plt.title('gradient')
plt.imshow(grad, cmap='gray')
plt.tight_layout()
plt.show()


circle_image = np.zeros((500, 500), np.uint8)
cv2.circle(circle_image, (250, 250), 100, 255, -1)

rect_image = np.zeros((500, 500), np.uint8)
cv2.rectangle(rect_image, (100, 100), (400, 250), 255, -1)

circle_and_rect_image = circle_image & rect_image

circle_or_rect_image = circle_image | rect_image

plt.figure(figsize=(10, 10))
plt.subplot(221)
plt.axis('off')
plt.title('circle')
plt.imshow(circle_image, cmap='gray')
plt.subplot(222)
plt.axis('off')
plt.title('rectangle')
plt.imshow(rect_image, cmap='gray')
plt.subplot(223)
plt.axis('off')
plt.title('circle & rectangle')
plt.imshow(circle_and_rect_image, cmap='gray')
plt.subplot(224)
plt.axis('off')
plt.title('circle | rectangle')
plt.imshow(circle_or_rect_image, cmap='gray')
plt.tight_layout()
plt.show()








