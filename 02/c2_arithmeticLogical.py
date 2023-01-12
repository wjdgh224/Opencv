import cv2, numpy as np

img5 = cv2.imread('data/img.jpg', cv2.IMREAD_GRAYSCALE)

mask = np.full(shape=img5.shape, fill_value=0, dtype=np.uint8)
#print(img5.shape)
h, w = img5.shape
x = (int)(w/2) - 60; y = (int)(h/2) - 60
cv2.rectangle(mask, (x,y), (x+120, y+120), (255,255,255), -1) # 움수는 사각 내부 흰색, 이부분 and시 살아 남음

ress = []
# ress.append(cv2.add(img1,img2))
# ress.append(cv2.addWeighted(img1, 0.5, img2, 0.5, 0))
# ress.append(cv2.subtract(img3, img4))
# ress.append(cv2.absdiff(img3, img4))
ress.append(cv2.bitwise_not(img5))
ress.append(cv2.bitwise_and(img5, mask))

cv2.imshow('not',ress[0])
cv2.imshow('and',ress[1])
cv2.waitKey()
cv2.destroyAllWindows()