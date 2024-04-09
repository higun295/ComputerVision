# 1. 입력 영상에 임의의 노이즈를 입힌다.
# 2. Gaussian Filtering 적용 후 결과 출력
# 3. Median Filtering 적용 후 결과 출력
# 4. Bilateral Filtering 적용 후 결과 출력
# 5. 각 결과에 대해 노이즈 입히기 전과 절대값 차이를 취해서 결과 출력
import cv2
import numpy as np
from matplotlib import pyplot as plt

image = cv2.imread('./data/Lena.png').astype(np.float32) / 255

noised = (image + 0.2 * np.random.rand(*image.shape).astype(np.float32))
noised = noised.clip(0, 1)
plt.imshow(noised[:, :, [2, 1, 0]])
plt.show()

