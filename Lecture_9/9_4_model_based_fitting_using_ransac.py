import cv2
import numpy as np
import matplotlib.pyplot as plt

# 이미지 불러오기
img0 = cv2.imread('./data/Lena.png', cv2.IMREAD_GRAYSCALE)
img1 = cv2.imread('./data/Lena_rotated.png', cv2.IMREAD_GRAYSCALE)

# ORB 특징점 검출기 생성
detector = cv2.ORB_create(100)
kps0, fea0 = detector.detectAndCompute(img0, None)
kps1, fea1 = detector.detectAndCompute(img1, None)

# 특징점 매칭 객체 생성
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
matches = matcher.match(fea0, fea1)

# 매칭된 점 집합 생성
pts0 = np.float32([kps0[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
pts1 = np.float32([kps1[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

# 호모그래피 계산
H, mask = cv2.findHomography(pts0, pts1, cv2.RANSAC, 3.0)

# 매칭 결과 시각화
plt.figure(figsize=(20, 10))
plt.subplot(211)
plt.axis('off')
plt.title('All Matches')
dbg_img_all = cv2.drawMatches(img0, kps0, img1, kps1, matches, None)
plt.imshow(cv2.cvtColor(dbg_img_all, cv2.COLOR_BGR2RGB))

plt.subplot(212)
plt.axis('off')
plt.title('Filtered Matches')
filtered_matches = [m for i, m in enumerate(matches) if mask[i]]
dbg_img_filtered = cv2.drawMatches(img0, kps0, img1, kps1, filtered_matches, None)
plt.imshow(cv2.cvtColor(dbg_img_filtered, cv2.COLOR_BGR2RGB))

plt.tight_layout()
plt.show()
