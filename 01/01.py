import cv2
import argparse
import os
import numpy as np
import random

# print(os.getcwd())

parser = argparse.ArgumentParser()
parser.add_argument('--path', default='data/img.png', help='Image path.')
params = parser.parse_args()
img = cv2.imread(params.path, cv2.IMREAD_GRAYSCALE)
image = cv2.imread('data/img.png')
w, h =image.shape[1], image.shape[0]


assert img is not None
print('read {}'.format(params.path))
print('shape:', img.shape)
print('dtype:', img.dtype)

#ori
print("original image shape:", img.shape)
#resize
width, height =  128, 256
resized_img = cv2.resize(img, (width, height))
print('resized to 128x256 image shape:', resized_img.shape)
#mul
w_mult, h_mult = 0.25, 0.5
resized_img = cv2.resize(img, (0,0), resized_img, w_mult, h_mult)
print('image shape:', resized_img.shape)
#interpolation
w_mult, h_mult = 2, 4
resized_img = cv2.resize(img, (0,0), resized_img, w_mult, h_mult, cv2.INTER_NEAREST)  # 기본: INTER_LINESAR
print('half sized image shape:', resized_img.shape)
#x축 뒤집기
img_flip_along_x = cv2.flip(img, 0)
#y
img_flip_along_y = cv2.flip(img, 1)
#x, y
img_flipped_xy = cv2.flip(img, -1)

#낮은 압축률 저장 : 디코딩 빠름
cv2.imwrite('data/img_compressed.png', img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
# 원본과 비교
#saved_img = cv2.imread(params.out_png) # 변수 없음?
#assert saved_img.all() == img.all()
#낮은 화질 저장
cv2.imwrite('data/img_compressed.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 0])


orig = cv2.imread('data/img.png')
orig_size = orig.shape[0:2]
cv2.imshow("Original image", orig)
cv2.waitKey(2000)


cv2.namedWindow('window')
fill_val = np.array([255,255,255], np.uint8)

def trackbar_callback(idx, value):
    fill_val[idx] = value

cv2.createTrackbar('R', 'window', 255, 255, lambda v: trackbar_callback(2, v))
cv2.createTrackbar('G', 'window', 255, 255, lambda v: trackbar_callback(1, v))
cv2.createTrackbar('B', 'window', 255, 255, lambda v: trackbar_callback(0, v))
import time
while True:
    image = np.full((500, 500, 3), fill_val) # 500x3배열 500개, 3열 fill_val로 채우기
    cv2.imshow('window', image)
    key = cv2.waitKey(3)
    print(image)
    time.sleep(2)
    if key==27:
        break

cv2.destroyAllWindows()


#2d
def rand_pt(mult=1.):
    return (random.randrange(int(w*mult)), random.randrange(int(h*mult)))

cv2.circle(image, rand_pt(), 40, (255, 0, 0))
cv2.circle(image, rand_pt(), 5, (255, 0, 0), cv2.FILLED)
cv2.circle(image, rand_pt(), 40, (255, 85, 85), 2)
cv2.circle(image, rand_pt(), 40, (255, 170, 170), 2, cv2.LINE_AA)

cv2.line(image, rand_pt(), rand_pt(), (0, 255, 0))
cv2.line(image, rand_pt(), rand_pt(), (85, 255, 85), 3)
cv2.line(image, rand_pt(), rand_pt(), (170, 255, 170), 3, cv2.LINE_AA)

cv2.arrowedLine(image, rand_pt(), rand_pt(), (0, 0, 255), 3, cv2.LINE_AA)

cv2.rectangle(image, rand_pt(), rand_pt(), (255, 255, 0), 3)

cv2.ellipse(image, rand_pt(), rand_pt(0.3), random.randrange(360), 0, 360, (255, 255, 255), 3)

cv2.putText(image, 'OpenCV', rand_pt(), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)

image_to_show = np.copy(image)

def rand_pt():
    return (random.randrange(w), random.randrange(h))

finish = False
while not finish:
    cv2.imshow('result', image_to_show)
    key = cv2.waitKey(0)
    if key == ord('p'):
        for pt in [rand_pt() for _ in range(10)]:
            cv2.circle(image_to_show, pt, 3, (255,0,0), -1)
    elif key == ord('l'):
        cv2.line(image_to_show, rand_pt(), rand_pt(), (0, 255, 0), 3)
    elif key == ord('r'):
        cv2.rectangle(image_to_show, rand_pt(), rand_pt(), (0, 0, 255), 3)
    elif key == ord('e'):
        cv2.ellipse(image_to_show, rand_pt(), rand_pt(), random.randrange(360), 0, 360, (255, 255, 0), 3)
    elif key == ord('t'):
        cv2.putText(image_to_show, 'OpenCV', rand_pt(), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
    elif key == ord('c'):
        image_to_show = np.copy(image)
    elif key == 27:
        finish = True
  
# 이미지 드래그 후 자르기
image_to_show = np.copy(image)

mouse_pressed = False
s_x  = s_y = e_x = e_y = -1

def mouse_callback(event, x, y, flags, param):
    global image_to_show, s_x, s_y, e_x, e_y, mouse_pressed

    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_pressed = True
        s_x, s_y = x, y
        image_to_show = np.copy(image)
    
    elif event == cv2.EVENT_MOUSEMOVE:
        if mouse_pressed:
            image_to_show = np.copy(image)
            cv2.rectangle(image_to_show, (s_x, s_y), (x, y), (0, 255, 0), 1)

    elif event == cv2.EVENT_LBUTTONUP:
        mouse_pressed = False
        e_x, e_y = x, y

cv2.namedWindow('image')
cv2.setMouseCallback('image', mouse_callback)

while True:
    cv2.imshow('image', image_to_show)
    k = cv2.waitKey(1)

    if k == ord('c'):
        if s_y > e_y:
            s_y, e_y = e_y, s_y
        if s_x > e_x:
            s_x, e_x = e_x, s_x

        if e_y - s_y > 1 and e_x - s_x > 0:
            image = image[s_y:e_y, s_x:e_x]
            image_to_show = np.copy(image)
        
    elif k == 27:
        break
cv2.destroyAllWindows()

