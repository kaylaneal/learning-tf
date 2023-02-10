import numpy as np
import cv2

img = cv2.imread("sundip.jpg", 1)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
res, thresh = cv2.threshold(hsv[:, :, 0], 73, 255, cv2.THRESH_BINARY)
#cv2.imshow("Thresh", thresh)

edges = cv2.Canny(img, 100, 200, apertureSize = 3)
#cv2.imshow("Canny", edges)

edge_inv = 255 - edges			# black instead of white

# Increase size of the "border"
kernel = np.ones((3, 3), 'uint8')
erode = cv2.erode(edge_inv, kernel, iterations = 1)

# Combine eroded and threshold image
canny_thresh = cv2.bitwise_and(erode, thresh)
cv2.imshow("Canny Thresh", canny_thresh)

# Contours
contours, hierarchy = cv2.findContours(canny_thresh, cv2.RETR_TREE,
							cv2.CHAIN_APPROX_SIMPLE)

objects = img.copy()

for c in contours:
	area = cv2.contourArea(c)

	if area < 15000:
		# too small to consider
		continue
	print("Area: {a}".format(a = area))

	cv2.drawContours(objects, [c], -1, (255, 255, 255), 1)

	M = cv2.moments(c)
	cx = int(M['m10'] / M['m00'])
	cy = int(M['m01'] / M['m00'])
	cv2.circle(objects, (cx, cy), 4, (255, 255, 0), -1)

cv2.imshow("Final Draw Over", objects)


cv2.waitKey(0)
cv2.destroyAllWindows()