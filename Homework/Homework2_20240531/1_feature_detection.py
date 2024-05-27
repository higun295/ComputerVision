# boat1, budapest1, newspaper1,s1 선택 후 Canny Edge와 Harris Corner를 검출해서 결과를 출력하는 코드 작성

import cv2
import numpy as np
from matplotlib import pyplot as plt

original_image = cv2.imread('./data/s1.jpg')
gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

# Canny Edge Detection
canny_edge_image = cv2.Canny(gray, 100, 200)

# Harris Corner Detection
harris_corner_image = original_image.copy()
gray_float = np.float32(gray)
dst = cv2.cornerHarris(gray_float, 2, 3, 0.04)
dst = cv2.dilate(dst, None)
harris_corner_image[dst > 0.01 * dst.max()] = [0, 0, 255]

plt.figure(figsize=(30, 15))
plt.subplot(1, 3, 1)
plt.title('Original Image', fontsize=25)
plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))

plt.subplot(1, 3, 2)
plt.title('Canny Edge Detection', fontsize=25)
plt.imshow(canny_edge_image, cmap='gray')

plt.subplot(1, 3, 3)
plt.title('Harris Corner Detection', fontsize=25)
plt.imshow(cv2.cvtColor(harris_corner_image, cv2.COLOR_BGR2RGB))

plt.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.05, wspace=0.1, hspace=0.1)
plt.show()
