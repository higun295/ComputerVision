# Stitching.zip에서 dog_a, dog_b 두 사진을 이용해서 Good feature to Track을 추출하고 Pyramid Lucas-Kanade 알고리즘을 적용해서 Optical Flow를 구하는 코드를 작성하시오.

import cv2
import numpy as np
import matplotlib.pyplot as plt

# 이미지 읽기
img1 = cv2.imread('./data/dog_a.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('./data/dog_b.jpg', cv2.IMREAD_GRAYSCALE)

# Good Features to Track 추출
features_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
p0 = cv2.goodFeaturesToTrack(img1, mask=None, **features_params)

# Pyramid Lucas-Kanade Optical Flow 설정
lk_params = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Optical Flow 계산
p1, st, err = cv2.calcOpticalFlowPyrLK(img1, img2, p0, None, **lk_params)

# 좋은 점들만 선택 (status가 1인 것들)
good_new = p1[st == 1]
good_old = p0[st == 1]

# 결과 이미지 생성
output_img = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

for i, (new, old) in enumerate(zip(good_new, good_old)):
    a, b = new.ravel()
    c, d = old.ravel()
    a, b, c, d = int(a), int(b), int(c), int(d)
    output_img = cv2.line(output_img, (a, b), (c, d), (0, 255, 0), 2)
    output_img = cv2.circle(output_img, (a, b), 5, (0, 0, 255), -1)

# 결과 출력
plt.figure(figsize=(10, 10))
plt.imshow(output_img)
plt.title('Optical Flow using Pyramid Lucas-Kanade', fontsize=25)
plt.axis('off')

plt.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.05, wspace=0.1, hspace=0.1)
plt.show()