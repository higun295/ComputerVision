import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('./Data/C_Key.jpg')

# 이미지 사이즈 해상도에 맞게 조정 필요
new_width = 1200
ratio = new_width / image.shape[1]
new_height = int(image.shape[0] * ratio)

# 이미지 크기 조정
resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
cv2.imshow("gray_image", gray_image)
cv2.waitKey(0)
cv2.destroyAllWindows()