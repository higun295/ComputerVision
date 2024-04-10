# 1. 입력 영상에 임의의 노이즈를 입힌다.
# 2. Gaussian Filtering 적용 후 결과 출력
# 3. Median Filtering 적용 후 결과 출력
# 4. Bilateral Filtering 적용 후 결과 출력
# 5. 각 결과에 대해 노이즈 입히기 전과 절대값 차이를 취해서 결과 출력

import cv2
import numpy as np
from matplotlib import pyplot as plt

image = cv2.imread('./data/Lena.png').astype(np.float32) / 255

# 1. 입력 영상에 임의의 노이즈를 입힌다.
noised = (image + 0.2 * np.random.rand(*image.shape).astype(np.float32))
noised = noised.clip(0, 1)
plt.imshow(noised[:, :, [2, 1, 0]])
plt.show()

# 2. Gaussian Filtering 적용 후 결과 출력
gauss_blur = cv2.GaussianBlur(noised, (7, 7), 0)
plt.imshow(gauss_blur[:, :, [2, 1, 0]])
plt.show()

# 3. Median Filtering 적용 후 결과 출력
median_blur = cv2.medianBlur((noised * 255).astype(np.uint8), 7)
plt.imshow(median_blur[:, :, [2, 1, 0]])
plt.show()

# 4. Bilateral Filtering 적용 후 결과 출력
bilat = cv2.bilateralFilter(noised, -1, 0.3, 10)
plt.imshow(bilat[:, :, [2, 1, 0]])
plt.show()


def calculate_difference_with_absolute(source, target, title):
    difference = cv2.absdiff(source, target)
    plt.imshow(difference[:, :, [2, 1, 0]])
    plt.title(title)
    plt.show()


# 5. 각 결과에 대해 노이즈 입히기 전과 절대값 차이를 취해서 결과 출력
calculate_difference_with_absolute(image, gauss_blur, 'Difference after Gaussian Filtering')

median_blur_float = median_blur.astype(np.float32) / 255
calculate_difference_with_absolute(image, median_blur_float, 'Difference after Median Filtering')
calculate_difference_with_absolute(image, bilat, 'Difference after Bilateral Filtering')
