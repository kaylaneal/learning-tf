# BASIC IMAGE OPERATIONS WITH OPENCV
import numpy as np
import cv2


# creates black image array
black = np.zeros([150, 200, 1], 'uint8')
cv2.imshow("Black", black)
print("Black Pixel: ", black[0, 0, :])

# creates almost black image array -- fills array with one, closer to 0 than 255 (white)
ones = np.ones([150, 200, 3], 'uint8')
cv2.imshow("Ones", ones)
print("Ones Pixel: ", ones[0, 0, :])

# creates white image array
white = np.ones([150, 200, 3], 'uint16')
white *= (2**16 -1) # makes one value the highest possible value to create white image array
cv2.imshow("White", white)
print("White Pixel: ", white[0, 0, :])

# creates blue image array
color = ones.copy()
color[:, :] = (255, 0, 0) # B,G,R FORMAT
cv2.imshow("Blue", color)
print("Blue Pixel: ", color[0, 0, :])

cv2.waitKey(0)
cv2.destroyAllWindows()