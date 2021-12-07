import cv2
import matplotlib.pyplot as plt

image = cv2.imread('Scarlett.jpg')
(b, g, r) = cv2.split(image)
'''
b = image[:, :, 0]
g = image[:, :, 1]
r = image[:, :, 2]
'''

image_copy = cv2.merge([b, g, r])

'''
cv2.imshow('image_copy', image_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

image_without_Blue = cv2.imread('Scarlett.jpg')
image_without_Blue[:, :, 0] = 0

image_without_Green = cv2.imread('Scarlett.jpg')
image_without_Green[:, :, 1] = 0

image_without_Red = cv2.imread('Scarlett.jpg')
image_without_Red[:, :, 2] = 0

(b_1, g_1, r_1) = cv2.split(image_without_Blue)
(b_2, g_2, r_2) = cv2.split(image_without_Green)
(b_3, g_3, r_3) = cv2.split(image_without_Red)


def show_with_matplotlib(color_img, title, pos):

    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(3, 6, pos)
    plt.imshow(img_RGB)
    plt.title(title, fontsize=8)
    plt.axis('off')


plt.figure(figsize=(13, 5))
plt.suptitle('splitting and merging channels in opencv', fontsize=12, fontweight='bold')

show_with_matplotlib(image, "BRG - image", 1)
show_with_matplotlib(cv2.cvtColor(b, cv2.COLOR_GRAY2BGR), "BGR - (B)", 2)
show_with_matplotlib(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR), "BGR - (G)", 2 + 6)
show_with_matplotlib(cv2.cvtColor(r, cv2.COLOR_GRAY2BGR), "BGR - (R)", 2 + 6 * 2)

show_with_matplotlib(image_copy, "BGR - image (merge)", 1 + 6)
show_with_matplotlib(image_without_Blue, "BGR without B", 3)
show_with_matplotlib(image_without_Green, "BGR without G", 3 + 6)
show_with_matplotlib(image_without_Red, "BGR without R", 3 + 6 * 2)

show_with_matplotlib(cv2.cvtColor(b_1, cv2.COLOR_GRAY2BGR), "BGR without B (B)", 4)
show_with_matplotlib(cv2.cvtColor(g_1, cv2.COLOR_GRAY2BGR), "BGR without B (G)", 4 + 6)
show_with_matplotlib(cv2.cvtColor(r_1, cv2.COLOR_GRAY2BGR), "BGR without B (R)", 4 + 6 * 2)

show_with_matplotlib(cv2.cvtColor(b_2, cv2.COLOR_GRAY2BGR), "BGR without R (B)", 5)
show_with_matplotlib(cv2.cvtColor(g_2, cv2.COLOR_GRAY2BGR), "BGR without R (G)", 5 + 6)
show_with_matplotlib(cv2.cvtColor(r_2, cv2.COLOR_GRAY2BGR), "BGR without R (R)", 5 + 6 * 2)

show_with_matplotlib(cv2.cvtColor(b_3, cv2.COLOR_GRAY2BGR), "BGR without R (B)", 6)
show_with_matplotlib(cv2.cvtColor(g_3, cv2.COLOR_GRAY2BGR), "BGR without R (G)", 6 + 6)
show_with_matplotlib(cv2.cvtColor(r_3, cv2.COLOR_GRAY2BGR), "BGR without R (R)", 6 + 6 * 2)

plt.show()