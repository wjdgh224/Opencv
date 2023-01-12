import cv2, numpy as np

img1 = cv2.imread('data/img.jpg', cv2.IMREAD_GRAYSCALE)

multi_lut = np.full(shape=[256], fill_value=0, dtype=np.uint8)
log_lut = np.full(shape=[256], fill_value=0, dtype=np.uint8)
invol_lut = np.full(shape=[256], fill_value=0, dtype=np.uint8)
sel_lut = np.full(shape=[256], fill_value=0, dtype=np.uint8)
multi_v = 2; gamma1 = 0.1; gamma2 = 0.6
thres1 = 5; thres2 = 100
max_v_log = 255/ np.log(1+255)
max_v_invol = 255/ np.power(255, gamma1)
max_v_sel = 100/ np.power(thres2, gamma2)


for i in range(256):
    val = i * multi_v
    if val > 255 : val = 255
    multi_lut[i] = val
    log_lut[i] = np.round(max_v_log * np.log(1+i))
    invol_lut[i] = np.round(max_v_invol * np.power(i, gamma1))
    if i < thres1 : sel_lut[i] = i
    elif i > thres2 : sel_lut[i] = i
    else: sel_lut[i] = np.round(max_v_sel * np.power(i, gamma2))
ress = []
ress.append(cv2.LUT(img1, multi_lut))
ress.append(cv2.LUT(img1, log_lut))
ress.append(cv2.LUT(img1, invol_lut))
ress.append(cv2.LUT(img1, sel_lut))

for i in range(len(ress)):
    cv2.imshow(str(i), ress[i])
cv2.waitKey()
cv2.destroyAllWindows()