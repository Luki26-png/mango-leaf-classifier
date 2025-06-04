import cv2
import numpy as np

#tentukan upper bound dan lower bound
lower_green = np.array([30, 40, 40])
upper_green = np.array([90, 255, 255])

#Masukkan gambar ke program
img = cv2.imread('sample.jpg')

#konversi ke hsv color space
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

#buat mask
mask = cv2.inRange(hsv, lower_green, upper_green)

#terapkan mask ke gambar asli
result = cv2.bitwise_and(img, img, mask=mask)


cv2.imwrite('sample/leaf_segmented.jpg',result)


cv2.imwrite('sample/leaf_mask.jpg',mask)