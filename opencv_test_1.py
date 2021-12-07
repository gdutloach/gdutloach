import cv2
from matplotlib import pyplot as plt
import numpy as np

img_opencv = cv2.imread('Scarlett.jpg')  # BRG格式
b, g, r = cv2.split(img_opencv)  # split语句耗时
'''
b = img_opencv[:,:,0]
g = img_opencv[:,:,1]    等价上一句
r = img_opencv[:,:,2]
'''
img_matplotlib = cv2.merge([r, g, b])  # Rbg格式
img_concats = np.concatenate((img_opencv, img_matplotlib), axis=1)  # 图片链接

top_left_corner = img_opencv[0:500, 0:500]
cv2.imshow('top_left_corner', top_left_corner)

img_opencv_shape = img_opencv.shape
img_opencv_size = img_opencv.size
img_opencv_dtype = img_opencv.dtype
print(img_opencv_shape)
print(img_opencv_size)
print(img_opencv_dtype)

cv2.imshow('bgr image and rgb image', img_concats)
cv2.imshow('brg image', img_opencv)
cv2.imshow('rgb image', img_matplotlib)
cv2.waitKey(0)
cv2.destroyAllWindows()

plt.subplot(121)
plt.imshow(img_opencv)
plt.subplot(122)
plt.imshow(img_matplotlib)
plt.show()
