from matplotlib import pyplot as plt
import cv2, numpy as np

img1 = cv2.imread('data/img.jpg', cv2.IMREAD_GRAYSCALE)

res1 = cv2.equalizeHist(img1)
ch1 = [0]; ranges1 = [0, 256]; histSize1 = [256]
hist1 = cv2.calcHist([img1], ch1, None, histSize1, ranges1)
hist2 = cv2.calcHist([res1], ch1, None, histSize1, ranges1)

multi_lut = np.full(shape=[256], fill_value=0, dtype=np.uint8)
log_lut = np.full(shape=[256], fill_value=0, dtype=np.uint8)
invol1_lut = np.full(shape=[256], fill_value=0, dtype=np.uint8)
multi_v = 2; gamma1 = 0.4
thres1 = 5; thres2 = 100
max_v_log = 255/ np.log(1+255)
max_v_invol1 = 255/ np.power(255, gamma1)

for i in range(256):
    val = i * multi_v
    if val > 255 : val = 255
    multi_lut[i] = val
    log_lut[i] = np.round(max_v_log * np.log(1+i))
    invol1_lut[i] = np.round(max_v_invol1 * np.power(i, gamma1))
   
res2 = cv2.LUT(img1, multi_lut)
res3 = cv2.LUT(img1, log_lut)
res4 = cv2.LUT(img1, invol1_lut)
hist3 = cv2.calcHist([res2], ch1, None, histSize1, ranges1)
hist4 = cv2.calcHist([res3], ch1, None, histSize1, ranges1)
hist5 = cv2.calcHist([res4], ch1, None, histSize1, ranges1)

bin_x1 = np.arange(256)

plt.title('Histogram')
plt.xlabel('Bin')
plt.ylabel('Frequency')
plt.plot(bin_x1, hist1, color='b')
plt.bar(bin_x1, hist2[:,0], width=6, color='r')
# plt.bar(bin_x1, hist3[:,0], width=6, color='g')
# plt.bar(bin_x1, hist4[:,0], width=6, color='b')
# plt.bar(bin_x1, hist5[:,0], width=6, color='r')
plt.show()