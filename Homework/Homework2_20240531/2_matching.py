# boat, budapest, newspaper, s1~s2 에서 두 장을 선택하고
# 각 영상에서 각각 SIFT, SURF, ORB를 추출한 후에 매칭 및 RANSAC을 통해서 두 장의 영상 간의 homography를 계산
# 이를 통해 한 장의 영상을 다른 한 장의 영상으로 warping하는 코드 작성

import cv2
import numpy as np

img1 = cv2.imread('./data/boat1.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('./data/boat2.jpg', cv2.IMREAD_GRAYSCALE)

def match_features(detector, img1, img2):
    keypoints1, descriptors1 = detector.detectAndCompute(img1, None)
    keypoints2, descriptors2 = detector.detectAndCompute(img2, None)

    # BFMatcher 생성
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)

    # 매칭된 특징점 좌표 추출
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 2)

    return src_pts, dst_pts, matches


# SIFT
sift = cv2.SIFT_create()
src_pts_sift, dst_pts_sift, matches_sift = match_features(sift, img1, img2)

# SURF
surf = cv2.xfeatures2d.SURF_create(400)
src_pts_surf, dst_pts_surf, matches_surf = match_features(surf, img1, img2)

# ORB
orb = cv2.ORB_create()
src_pts_orb, dst_pts_orb, matches_orb = match_features(orb, img1, img2)


# RANSAC을 통한 homography 계산
def calculate_homography(src_pts, dst_pts):
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    return H


H_sift = calculate_homography(src_pts_sift, dst_pts_sift)
H_surf = calculate_homography(src_pts_surf, dst_pts_surf)
H_orb = calculate_homography(src_pts_orb, dst_pts_orb)


# 이미지 warping
def warp_image(img, H, shape):
    return cv2.warpPerspective(img, H, shape)


height, width = img2.shape
warped_img_sift = warp_image(img1, H_sift, (width, height))
warped_img_surf = warp_image(img1, H_surf, (width, height))
warped_img_orb = warp_image(img1, H_orb, (width, height))