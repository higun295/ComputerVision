import cv2
import numpy as np
import matplotlib.pyplot as plt

# 이미지 경로
image_paths = [
    './cbnu_images/4_1.jpg', './cbnu_images/4_2.jpg', './cbnu_images/4_3.jpg',
    './cbnu_images/4_4.jpg', './cbnu_images/4_5.jpg'
]

# 이미지 불러오기 및 그레이스케일 변환
images = [cv2.imread(img_path) for img_path in image_paths]
gray_images = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in images]

# 이미지 크기 조정 (0.5배)
scale_factor = 0.5
images = [cv2.resize(img, (int(img.shape[1] * scale_factor), int(img.shape[0] * scale_factor))) for img in images]
gray_images = [cv2.resize(img, (int(img.shape[1] * scale_factor), int(img.shape[0] * scale_factor))) for img in
               gray_images]

# SIFT 특징점 검출기 생성
sift = cv2.SIFT_create()

# 기준 이미지
base_img = images[0]
base_gray = gray_images[0]

# 특징점과 디스크립터 추출
kp_base, des_base = sift.detectAndCompute(base_gray, None)

# 매칭 객체 생성
bf = cv2.BFMatcher(cv2.NORM_L2)

# 유사도 계산 및 매칭 시각화
similarity_scores = []

for i in range(1, len(images)):
    kp, des = sift.detectAndCompute(gray_images[i], None)
    matches = bf.knnMatch(des_base, des, k=2)

    # Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # 유사도 저장
    similarity_scores.append(len(good_matches))

    # 매칭된 점 집합 생성
    pts_base = np.float32([kp_base[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts_i = np.float32([kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # RANSAC을 사용한 호모그래피 계산
    H, mask = cv2.findHomography(pts_i, pts_base, cv2.RANSAC, 5.0)

    # 호모그래피를 사용한 이미지 변환
    height, width, channels = base_img.shape
    warped_img = cv2.warpPerspective(images[i], H, (width, height))

    # 기준 이미지와 변환된 이미지 블렌딩
    base_img = cv2.addWeighted(base_img, 0.5, warped_img, 0.5, 0)

    # 매칭 결과 시각화 (상위 100개 매칭만)
    num_matches_to_show = min(100, len(good_matches))
    dbg_img = cv2.drawMatches(images[0], kp_base, images[i], kp, good_matches[:num_matches_to_show], None,
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.figure(figsize=(20, 10))
    plt.imshow(cv2.cvtColor(dbg_img, cv2.COLOR_BGR2RGB))
    plt.title(f'4_1 vs 4_{i + 1} Matches')
    plt.axis('off')
    plt.show()

# 최종 겹쳐진 이미지 시각화
plt.figure(figsize=(20, 10))
plt.imshow(cv2.cvtColor(base_img, cv2.COLOR_BGR2RGB))
plt.title('Overlayed Image')
plt.axis('off')
plt.show()

# 유사도 그래프 시각화
plt.figure(figsize=(10, 5))
plt.bar(range(2, len(images) + 1), similarity_scores, color='blue')
plt.xlabel('Image Index')
plt.ylabel('Number of Good Matches')
plt.title('Similarity Scores between 4_1 and Other Images')
plt.show()
