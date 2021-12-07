import cv2
import matplotlib.pyplot as plt
import numpy as np


def show_with_matplotlib(color_img, title, pos):
    img_rgb = color_img[:, :, ::-1]

    ax = plt.subplot(2, 4, pos)
    plt.imshow(img_rgb)
    plt.title(title, fontsize=8)
    plt.axis('off')


image = cv2.imread('D:/python/practice/picture/calss.jpg')


# 使用缩放因子
dst_image_1 = cv2.resize(image, None, fx=0.1, fy=0.1, interpolation=cv2.INTER_NEAREST)
dst_image_2 = cv2.resize(image, None, fx=0.1, fy=0.1, interpolation=cv2.INTER_LINEAR)
dst_image_3 = cv2.resize(image, None, fx=0.1, fy=0.1, interpolation=cv2.INTER_AREA)
dst_image_4 = cv2.resize(image, None, fx=0.1, fy=0.1, interpolation=cv2.INTER_CUBIC)
dst_image_5 = cv2.resize(image, None, fx=0.1, fy=0.1, interpolation=cv2.INTER_LANCZOS4)

# 旋转图像
dst_image_7 = image.copy()
height, width = image.shape[:2]
N = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), 180, 1)
spin_image = cv2.warpAffine(dst_image_7, N, (height, width))

dst_image_8 = image.copy()
height, width = image.shape[:2]
N = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), 180, 1)
spin_image_1 = cv2.warpAffine(dst_image_8, N, (width, height))  # 注意高度和宽度对应

# 平移图像
dst_image_6 = image.copy()
M = np.float32([[1, 0, 100], [0, 1, 30]])
width, height = image.shape[:2]
panning_image = cv2.warpAffine(dst_image_6, M, (width, height))

show_with_matplotlib(dst_image_1, "NEAREST - image", 1)
show_with_matplotlib(dst_image_2, "LINEAR - image", 2)
show_with_matplotlib(dst_image_3, "AREA - image", 3)
show_with_matplotlib(dst_image_4, "CUBIC - image", 4)
show_with_matplotlib(dst_image_5, "LANCZOS4 - image", 1 + 4)
show_with_matplotlib(panning_image, "panning - image", 2 + 4)
show_with_matplotlib(spin_image, "spin - image", 3 + 4)
show_with_matplotlib(spin_image_1, "spin_1 - image", 4 + 4)

plt.show()
