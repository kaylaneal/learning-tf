# BASIC IMAGE MANIPULATION:
import numpy as np
import cv2

# COLOR CHANNELS/IMAGE TYPES

dippy = cv2.imread("dippy.jpg", 1) 							# 1 signifies color image
cv2.imshow("Dippy", dippy)
cv2.moveWindow("Dippy", 0, 0)  								# image will open in top left hand corner
print("Dippy Image Shape: ", dippy.shape)
height, width, channels = dippy.shape

## VISUALIZE INDIVIDUAL CHANNELS
b, g, r = cv2.split(dippy)

rgb_split = np.empty([height, width*3, 3], 'uint8')

rgb_split[:, 0:width] = cv2.merge([b, b, b]) 				# blue channel on left side
rgb_split[:, width:width*2] = cv2.merge([g, g, g]) 			# green channel in middle
rgb_split[:, width*2:width*3] = cv2.merge([r, r, r]) 		# red channel on right side

cv2.imshow("Channels", rgb_split)
cv2.moveWindow("Channels", 0, height)

hsv = cv2.cvtColor(dippy, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv)
hsv_split = np.concatenate((h, s, v))
cv2.imshow("HSV Channels", hsv_split)


cv2.waitKey(0)
cv2.destroyAllWindows()
