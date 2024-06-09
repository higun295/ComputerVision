import cv2
import numpy as np
import matplotlib.pyplot as plt

# 이미지 불러오기
image_path1 = './data/piano_1.jpg'
image_path2 = './data/piano_with_pen_1.jpg'

image1 = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)

# 이미지를 동일한 크기로 맞추기 (크기가 다를 경우)
if image1.shape != image2.shape:
    image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))

# SIFT 특징점 검출기 생성
sift = cv2.SIFT_create()

# 특징점 및 디스크립터 검출
keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
keypoints2, descriptors2 = sift.detectAndCompute(image2, None)

# FLANN 기반 매칭 객체 생성
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

# 매칭
matches = flann.knnMatch(descriptors1, descriptors2, k=2)

# 좋은 매칭점만 선택 (Lowe's ratio test)
good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)

# 매칭된 결과 시각화
image_matches = cv2.drawMatches(image1, keypoints1, image2, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# RANSAC을 사용하여 호모그래피 행렬 계산
if len(good_matches) > 4:
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    height, width = image1.shape
    image2_aligned = cv2.warpPerspective(image2, H, (width, height))
else:
    raise ValueError("Not enough matches found to compute homography.")

# 눌린 건반 검출
# 이미지 정렬 후 차이 계산
diff_image = cv2.absdiff(image1, image2_aligned)
_, thresh = cv2.threshold(diff_image, 30, 255, cv2.THRESH_BINARY)

# 결과 시각화
plt.figure(figsize=(15, 7))
plt.subplot(1, 3, 1)
plt.imshow(image1, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(image2_aligned, cmap='gray')
plt.title('Aligned Image with Pressed Keys')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(thresh, cmap='gray')
plt.title('Detected Pressed Keys')
plt.axis('off')

plt.show()