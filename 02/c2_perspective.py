import cv2, numpy as np

img1 = cv2.imread('data/img.jpg')

h, w = img1.shape
point1_src = np.float32([[1, 1], [w-10, 10], [5, h-5], [w-4, h-4]])
point1_dst = np.float32([[15, 15], [w-10, 10], [5, h-5], [w-4, h-4]])
point2_src = np.float32([[1, 1], [w-10, 10], [5, h-5], [w-4, h-4]])
point2_dst = np.float32([[1, 1], [w-10, 10], [5, h-5], [w-4, h-4]])
