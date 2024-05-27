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

# 이미지가 제대로 읽혔는지 확인
if img1 is None or img2 is None:
    print("이미지를 읽어오는데 문제가 발생했습니다.")
    exit()

# SIFT 특징 추출
sift = cv2.SIFT_create()
keypoints1_sift, descriptors1_sift = sift.detectAndCompute(img1, None)
keypoints2_sift, descriptors2_sift = sift.detectAndCompute(img2, None)

# SURF 특징 추출 (hessianThreshold 값을 조정하여 검출할 특징점의 개수를 조절할 수 있음)
surf = cv2.xfeatures2d.SURF_create(400)
keypoints1_surf, descriptors1_surf = surf.detectAndCompute(img1, None)
keypoints2_surf, descriptors2_surf = surf.detectAndCompute(img2, None)

# ORB 특징 추출
orb = cv2.ORB_create()
keypoints1_orb, descriptors1_orb = orb.detectAndCompute(img1, None)
keypoints2_orb, descriptors2_orb = orb.detectAndCompute(img2, None)

# 특징점을 이미지에 그리기
img1_sift = cv2.drawKeypoints(img1, keypoints1_sift, None, (255, 0, 0), 4)
img2_sift = cv2.drawKeypoints(img2, keypoints2_sift, None, (255, 0, 0), 4)
img1_surf = cv2.drawKeypoints(img1, keypoints1_surf, None, (255, 0, 0), 4)
img2_surf = cv2.drawKeypoints(img2, keypoints2_surf, None, (255, 0, 0), 4)
img1_orb = cv2.drawKeypoints(img1, keypoints1_orb, None, (255, 0, 0), 4)
img2_orb = cv2.drawKeypoints(img2, keypoints2_orb, None, (255, 0, 0), 4)

# 결과 이미지 출력
plt.figure(figsize=(20, 10))

plt.subplot(2, 3, 1)
plt.imshow(img1_sift)
plt.title('SIFT Keypoints in Image 1')

plt.subplot(2, 3, 2)
plt.imshow(img1_surf)
plt.title('SURF Keypoints in Image 1')

plt.subplot(2, 3, 3)
plt.imshow(img1_orb)
plt.title('ORB Keypoints in Image 1')

plt.subplot(2, 3, 4)
plt.imshow(img2_sift)
plt.title('SIFT Keypoints in Image 2')

plt.subplot(2, 3, 5)
plt.imshow(img2_surf)
plt.title('SURF Keypoints in Image 2')

plt.subplot(2, 3, 6)
plt.imshow(img2_orb)
plt.title('ORB Keypoints in Image 2')

plt.show()