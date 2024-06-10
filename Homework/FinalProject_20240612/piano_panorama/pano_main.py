import cv2
import numpy as np
import matplotlib.pyplot as plt

# 이미지 불러오기
img1_path = './panorama_data/piano_1.jpg'
img2_path = './panorama_data/piano_2.jpg'

img1 = cv2.imread(img1_path)
img2 = cv2.imread(img2_path)

# 이미지 전처리 - Canny Edge Detection을 사용하여 대비 높이기
def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return edges

img1_edges = preprocess_image(img1)
img2_edges = preprocess_image(img2)

# SIFT 객체 생성 (특징점 민감도 조정)
sift = cv2.SIFT_create(contrastThreshold=0.04, edgeThreshold=10)

# 특징점과 디스크립터 추출
kp1, des1 = sift.detectAndCompute(img1_edges, None)
kp2, des2 = sift.detectAndCompute(img2_edges, None)

# 매칭 객체 생성
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = bf.match(des1, des2)

# 매칭 결과 시각화 함수
def draw_matches(img1, kp1, img2, kp2, matches, title):
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.figure(figsize=(20, 10))
    plt.imshow(img_matches)
    plt.title(title)
    plt.show()

# 매칭 결과 시각화
draw_matches(img1, kp1, img2, kp2, matches, 'SIFT Matches with Edge Detection')

# RANSAC을 사용하여 이상치 제거 및 호모그래피 계산 함수
def find_homography_ransac(kp1, kp2, matches):
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    return H, mask

# 호모그래피 계산 및 시각화
H, mask = find_homography_ransac(kp1, kp2, matches)
matches_mask = mask.ravel().tolist()

# 호모그래피 적용 결과 시각화 함수
def visualize_homography(img1, img2, H, title):
    h, w, _ = img1.shape
    img1_warp = cv2.warpPerspective(img1, H, (w + img2.shape[1], h))
    plt.figure(figsize=(20, 10))
    plt.subplot(121), plt.imshow(img2), plt.title('Base Image')
    plt.subplot(122), plt.imshow(img1_warp), plt.title(title)
    plt.show()

# 호모그래피 적용 결과 시각화
visualize_homography(img1, img2, H, 'SIFT Warped Image with Edge Detection')