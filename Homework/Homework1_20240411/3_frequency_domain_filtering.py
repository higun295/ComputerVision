# 1. 입력 영상을 DFT를 통해 주파수 도메인으로 변경.
# 2. 주파수 도메인으로 변경한 후 출력
# 3. 사용자로부터 반지름 정보 2개 입력 받기.
# 4. 영상의 중심을 원의 중심으로 하여 2개의 원 그리기.
# 5. 두 원 사이의 영역을 통과시키는 Band Pass 필터 구현.
# 6. 필터링 결과를 출력.

import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('./data/Lena.png', 0).astype(np.float32) / 255

# 1. 입력 영상을 DFT를 통해 주파수 도메인으로 변경.
fft = cv2.dft(image, flags=cv2.DFT_COMPLEX_OUTPUT)
shifted = np.fft.fftshift(fft, axes=[0, 1])

magnitude = cv2.magnitude(shifted[:, :, 0], shifted[:, :, 1])
magnitude = np.log(magnitude + 1)

# 2. 주파수 도메인으로 변경한 후 출력
plt.figure(figsize=(6, 6))
plt.axis('off')
plt.imshow(magnitude, cmap='gray')
plt.tight_layout()
plt.show()

# 3. 사용자로부터 반지름 정보 2개 입력 받기.
r_inner = int(input("내부 원의 반지름을 입력하세요: "))
r_outer = int(input("외부 원의 반지름을 입력하세요: "))

# 4. 영상의 중심을 원의 중심으로 하여 2개의 원 그리기.
rows, cols = image.shape
crow, ccol = rows // 2, cols // 2
mask = np.zeros((rows, cols, 2), np.uint8)
x, y = np.ogrid[:rows, :cols]
center_distance = (x - crow)**2 + (y - ccol)**2
mask[(center_distance >= r_inner**2) & (center_distance <= r_outer**2)] = 1

# 5. 두 원 사이의 영역을 통과시키는 Band Pass 필터 구현.
filtered = shifted * mask
filtered_shift = np.fft.ifftshift(filtered)
restored = cv2.idft(filtered_shift, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)

# 6. 필터링 결과를 출력.
plt.figure(figsize=(6, 6))
plt.imshow(restored, cmap='gray')
plt.axis('off')
plt.show()