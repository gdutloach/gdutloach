import numpy as np
import cv2


image = cv2.imread('Scarlett.jpg')
height, width = image.shape[:2]
'''
pts_1 = np.float32([[135, 45], [385, 45], [135, 230]])
pts_2 = np.float32([[135, 45], [385, 45], [150, 230]])
M = cv2.getAffineTransform(pts_1, pts_2)
dst_image = cv2.warpAffine(image, M, (width, height))
'''

pts_1 = np.float32([[450, 65], [517, 65], [431, 164], [552, 164]])
pts_2 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])
M = cv2.getPerspectiveTransform(pts_1, pts_2)
dst_image = cv2.warpPerspective(image, M, (300, 300))

cv2.namedWindow('dst_image', cv2.WINDOW_NORMAL)
cv2.imshow('dst_image', dst_image)
cv2.waitKey(0)
cv2.destroyAllWindows()