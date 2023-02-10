# OBJECT DETECTION WITH OPENCV

import cv2
import numpy as np

bw = cv2.imread("sundip.jpg", 0)			# 0 denotes Black and White image
height, width = bw.shape[0:2]				# exclude channels
cv2.imshow("Original BW", bw)

# Thresholding Model:
## goal: extract objects --> all objects = 1, everything else = 0

binary = np.zeros([height, width, 1], 'uint8')		# max value 255 (white) min value 0 (black)

threshold = 85

## compare every  pixel to this threshold, assign appropriately
'''
for row in range(0, height):
	for col in range(0, width):
		if bw[row][col] > threshold:
			binary[row][col] = 255

cv2.imshow("Slow Binary", binary)
'''

ret, thresh = cv2.threshold(bw, threshold, 255, cv2.THRESH_BINARY)
cv2.imshow("CV Threshold", thresh)

thresh_adapt = cv2.adaptiveThreshold(bw, 255, 
									cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
									cv2.THRESH_BINARY, 115, 1)
cv2.imshow("CV Adaptive", thresh_adapt)

cv2.waitKey(0)
cv2.destroyAllWindows()

