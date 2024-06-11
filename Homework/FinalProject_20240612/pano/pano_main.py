import cv2
import numpy as np
import matplotlib.pyplot as plt

# 이미지 경로
img1_path = './data/1_1.jpg'
img2_path = './data/1_3.jpg'

# 이미지 불러오기
img1 = cv2.imread(img1_path)
img2 = cv2.imread(img2_path)

# 이미지 크기 조정 (0.5배)
scale_factor = 0.5
img1 = cv2.resize(img1, (int(img1.shape[1] * scale_factor), int(img1.shape[0] * scale_factor)))
img2 = cv2.resize(img2, (int(img2.shape[1] * scale_factor), int(img2.shape[0] * scale_factor)))

# 그레이스케일로 변환
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# SIFT 특징점 검출기 생성
sift = cv2.SIFT_create()

# 특징점과 디스크립터 추출
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

# 특징점 매칭 객체 생성
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = bf.match(des1, des2)

# 매칭된 점들을 거리 기준으로 정렬
matches = sorted(matches, key=lambda x: x.distance)

# 상위 200개의 매칭 선택
top_matches = matches[:200]

# 매칭된 점 집합 생성
pts1 = np.float32([kp1[m.queryIdx].pt for m in top_matches]).reshape(-1, 1, 2)
pts2 = np.float32([kp2[m.trainIdx].pt for m in top_matches]).reshape(-1, 1, 2)

# RANSAC을 사용한 호모그래피 계산
H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
matches_mask = mask.ravel().tolist()

# 호모그래피를 사용한 이미지 변환
height, width, channels = img2.shape
warped_img1 = cv2.warpPerspective(img1, H, (width + img1.shape[1], height))

# 원본 이미지와 변환된 이미지를 결합하여 파노라마 생성
warped_img1[0:img2.shape[0], 0:img2.shape[1]] = img2

# 결과 시각화 (파노라마)
plt.figure(figsize=(20, 10))
plt.imshow(cv2.cvtColor(warped_img1, cv2.COLOR_BGR2RGB))
plt.title('Panorama')
plt.axis('off')
plt.show()

# 상위 200개 매칭 결과 시각화
dbg_img_top = cv2.drawMatches(img1, kp1, img2, kp2, top_matches, None, matchesMask=matches_mask, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.figure(figsize=(20, 10))
plt.imshow(cv2.cvtColor(dbg_img_top, cv2.COLOR_BGR2RGB))
plt.title('Top 200 Matches')
plt.axis('off')
plt.show()

# RANSAC을 사용하여 필터링된 매칭 결과 시각화
filtered_matches = [m for i, m in enumerate(top_matches) if matches_mask[i]]
dbg_img_filtered = cv2.drawMatches(img1, kp1, img2, kp2, filtered_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.figure(figsize=(20, 10))
plt.imshow(cv2.cvtColor(dbg_img_filtered, cv2.COLOR_BGR2RGB))
plt.title('Filtered Matches (RANSAC Inliers)')
plt.axis('off')
plt.show()
