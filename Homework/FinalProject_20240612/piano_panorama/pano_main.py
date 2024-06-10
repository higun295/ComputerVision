import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt

# 이미지 불러오기
image_files = glob.glob('./piano_panorama/panorama_data/*.jpg')
images = [cv2.imread(img) for img in image_files]

# SIFT 객체 생성
sift = cv2.SIFT_create()

# 모든 이미지에 대해 특징점과 디스크립터 추출
keypoints_descriptors = [sift.detectAndCompute(image, None) for image in images]

# 매칭 객체 생성
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

# 모든 이미지 쌍에 대해 매칭 수행
matches = []
for i in range(len(images) - 1):
    kp1, des1 = keypoints_descriptors[i]
    kp2, des2 = keypoints_descriptors[i + 1]
    matches.append(bf.match(des1, des2))

# 매칭 결과 시각화
def draw_matches(img1, kp1, img2, kp2, matches):
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.figure(figsize=(20, 10))
    plt.imshow(img_matches)
    plt.show()

# 첫 번째 이미지 쌍에 대해 매칭 결과 시각화
kp1, des1 = keypoints_descriptors[0]
kp2, des2 = keypoints_descriptors[1]
draw_matches(images[0], kp1, images[1], kp2, matches[0])
