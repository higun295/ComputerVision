# boat1, budapest1, newspaper1,s1 선택 후 Canny Edge와 Harris Corner를 검출해서 결과를 출력하는 코드 작성

import cv2
import numpy as np
from matplotlib import pyplot as plt

# 이미지 경로
image_path = './data/boat1.jpg'

# 이미지 로드
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Canny Edge Detection
edges = cv2.Canny(gray, 100, 200)

# Harris Corner Detection
gray_float = np.float32(gray)
dst = cv2.cornerHarris(gray_float, 2, 3, 0.04)
dst = cv2.dilate(dst, None)  # 결과를 확실하게 하기 위해 dilation 적용
image[dst > 0.01 * dst.max()] = [0, 0, 255]  # 빨간색으로 코너 표시

# 결과 출력
plt.figure(figsize=(20, 10))

plt.subplot(1, 3, 1)
plt.title('Original Image')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

plt.subplot(1, 3, 2)
plt.title('Canny Edge Detection')
plt.imshow(edges, cmap='gray')

plt.subplot(1, 3, 3)
plt.title('Harris Corner Detection')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

plt.show()
