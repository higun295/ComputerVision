import cv2
import numpy as np

img = cv2.imread('./data/scenetext01.jpg', cv2.IMREAD_COLOR)

cv2.imshow('test', img)
cv2.waitKey()
corners = cv2.cornerHarris(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 2, 3, 0.04)

corners = cv2.dilate(corners, None)

show_img = np.copy(img)
show_img[corners > 0.1 * corners.max()] = [0, 0, 255]

# corners = cv2.nor