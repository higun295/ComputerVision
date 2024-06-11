import cv2
import numpy as np
import matplotlib.pyplot as plt

# 이미지 경로
img1_path = './cbnu_images/1.jpg'
img2_path = './cbnu_images/3.jpg'

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

# KNN 매칭과 Lowe's ratio test 적용
bf = cv2.BFMatcher(cv2.NORM_L2)
matches = bf.knnMatch(des1, des2, k=2)

# Lowe's ratio test
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

# 매칭된 점 집합 생성
pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# RANSAC을 사용한 호모그래피 계산
H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
matches_mask = mask.ravel().tolist()

# 호모그래피를 사용한 이미지 변환
height, width, channels = img2.shape
warped_img1 = cv2.warpPerspective(img1, H, (width + img1.shape[1], height))


# 이미지 크기 조정 함수
def resize_to_match(img1, img2):
    height1, width1 = img1.shape[:2]
    height2, width2 = img2.shape[:2]

    if height1 != height2 or width1 != width2:
        img2_resized = cv2.resize(img2, (width1, height1))
        return img1, img2_resized
    return img1, img2


# 이미지 블렌딩을 사용하여 파노라마 생성
def blend_images(img1, img2):
    img1, img2 = resize_to_match(img1, img2)
    mask1 = np.zeros((img1.shape[0], img1.shape[1]), dtype=np.uint8)
    mask1[:, :img2.shape[1]] = 255
    mask2 = 255 - mask1
    img1_float = img1.astype(np.float32)
    img2_float = img2.astype(np.float32)
    blended_img = cv2.addWeighted(img1_float, 0.5, img2_float, 0.5, 0)
    return blended_img.astype(np.uint8)


blended_img = blend_images(warped_img1, img2)

# 결과 시각화
plt.figure(figsize=(20, 10))

plt.subplot(131)
plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
plt.title('Image 1')
plt.axis('off')

plt.subplot(132)
plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
plt.title('Image 2')
plt.axis('off')

plt.subplot(133)
plt.imshow(cv2.cvtColor(blended_img, cv2.COLOR_BGR2RGB))
plt.title('Panorama')
plt.axis('off')

plt.tight_layout()
plt.show()

# 매칭 결과 시각화
dbg_img_all = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, matchesMask=matches_mask,
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.figure(figsize=(20, 10))
plt.imshow(cv2.cvtColor(dbg_img_all, cv2.COLOR_BGR2RGB))
plt.title('Filtered Matches')
plt.axis('off')
plt.show()
