# BASIC IMAGE MANIPULATION:
import numpy as np
import cv2


# BLUR, SCALE, ROTATE, DILATE, ETC.

image = cv2.imread("roosevelts.jpg")
cv2.imshow("Original", image)

blur = cv2.GaussianBlur(image, (5, 55), 0)					# (5, 55) represents Gaussian kernel size; must be positive and odd
cv2.imshow("Blur", blur)

kernel = np.ones((5, 5), 'uint8')

dilate = cv2.dilate(image, kernel, iterations = 1)
erode = cv2.erode(image, kernel, iterations = 1)

cv2.imshow("Dilate", dilate)
cv2.imshow("Erode", erode)

img_half = cv2.resize(image, (0, 0), fx = 0.5, fy = 0.5)		# image is half the size of the original in both directions
img_stretch = cv2.resize(image, (600, 600))
img_stretch_near = cv2.resize(image, (600, 600), interpolation = cv2.INTER_NEAREST)

cv2.imshow("Half", img_half)
cv2.imshow("Stretch", img_stretch)
cv2.imshow("Stretch Near", img_stretch_near)

M = cv2.getRotationMatrix2D((0,0), -30, 1)
rotated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
cv2.imshow("Rotated", rotated)

cv2.waitKey(0)
cv2.destroyAllWindows()