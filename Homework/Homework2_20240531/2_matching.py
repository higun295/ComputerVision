# boat, budapest, newspaper, s1~s2 에서 두 장을 선택하고
# 각 영상에서 각각 SIFT, SURF, ORB를 추출한 후에 매칭 및 RANSAC을 통해서 두 장의 영상 간의 homography를 계산
# 이를 통해 한 장의 영상을 다른 한 장의 영상으로 warping하는 코드 작성

import cv2
import matplotlib.pyplot as plt

# 이미지 경로
img1_path = './data/boat1.jpg'
img2_path = './data/boat2.jpg'

# 이미지 읽기
img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

# SIFT 생성
sift = cv2.SIFT_create()

# SIFT 특징점 및 디스크립터 추출
keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

# BFMatcher 생성
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

# 디스크립터 매칭
matches = bf.match(descriptors1, descriptors2)

# 매칭 결과 정렬 (거리가 가까운 순서로)
matches = sorted(matches, key=lambda x: x.distance)

# 매칭 결과 이미지 그리기
img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# 결과 이미지 출력
plt.figure(figsize=(20, 10))
plt.imshow(img_matches)
plt.title('SIFT Feature Matching')
plt.show()
