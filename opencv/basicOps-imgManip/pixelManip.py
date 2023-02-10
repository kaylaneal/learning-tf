# BASIC IMAGE MANIPULATION:
import numpy as np
import cv2

# PIXEL MANIPULATION/FILTERING

frank = cv2.imread("frank.jpg", 1)

gray = cv2.cvtColor(frank, cv2.COLOR_RGB2GRAY)
cv2.imwrite("gray.jpg", gray)								# saves image (new) as grayscale

fb = frank[:, :, 0]
fg = frank[:, :, 1]
fr = frank[:, :, 2]

rgba = cv2.merge((fb, fg, fr, fg))							# add an alpha channel (transparency) -- g means low/non green values are transparent
cv2.imwrite("rgba.png", rgba)
