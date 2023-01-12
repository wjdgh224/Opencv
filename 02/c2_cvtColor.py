import cv2

img1 = cv2.imread('data/img.jpg', cv2.IMREAD_UNCHANGED)

res1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV) # HSV

res1_split = cv2.split(res1) # H S V 분리
res1_split[2] = cv2.add(res1_split[2], 100) # V 조정
res1_merge = cv2.merge(res1_split)
res1_merge = cv2.cvtColor(res1_merge, cv2.COLOR_HSV2BGR) # V 조정후 BGR