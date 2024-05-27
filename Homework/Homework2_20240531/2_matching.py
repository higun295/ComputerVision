# boat, budapest, newspaper, s1~s2 에서 두 장을 선택하고
# 각 영상에서 각각 SIFT, SURF, ORB를 추출한 후에 매칭 및 RANSAC을 통해서 두 장의 영상 간의 homography를 계산
# 이를 통해 한 장의 영상을 다른 한 장의 영상으로 warping하는 코드 작성

import cv2
import numpy as np
from matplotlib import pyplot as plt


def draw_matches(img1, kp1, img2, kp2, matches):
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out_img = np.zeros((max([rows1, rows2]), cols1 + cols2, 3), dtype='uint8')
    out_img[:rows1, :cols1, :] = np.dstack([img1])
    out_img[:rows2, cols1:, :] = np.dstack([img2])

    for match in matches:
        img1_idx = match.queryIdx
        img2_idx = match.trainIdx

        (x1, y1) = kp1[img1_idx].pt
        (x2, y2) = kp2[img2_idx].pt

        cv2.circle(out_img, (int(x1), int(y1)), 4, (255, 0, 0), 1)
        cv2.circle(out_img, (int(x2) + cols1, int(y2)), 4, (255, 0, 0), 1)
        cv2.line(out_img, (int(x1), int(y1)), (int(x2) + cols1, int(y2)), (255, 0, 0), 1)

    return out_img


def match_features(img1, img2, detector):
    keypoints1, descriptors1 = detector.detectAndCompute(img1, None)
    keypoints2, descriptors2 = detector.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)

    return keypoints1, keypoints2, matches


def find_homography_and_warp(img1, img2, keypoints1, keypoints2, matches):
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    h, w = img1.shape[:2]
    img1_warp = cv2.warpPerspective(img1, M, (w, h))

    return img1_warp, M, mask


# 이미지 경로
image1_path = './data/boat1.jpg'
image2_path = './data/boat2.jpg'

# 이미지 로드
img1 = cv2.imread(image1_path)
img2 = cv2.imread(image2_path)

# 그레이스케일로 변환
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# SIFT 특징 추출
sift = cv2.SIFT_create()
keypoints1, keypoints2, matches = match_features(gray1, gray2, sift)
img1_warp_sift, M_sift, mask_sift = find_homography_and_warp(img1, img2, keypoints1, keypoints2, matches)
matched_img_sift = draw_matches(img1, keypoints1, img2, keypoints2, matches)

# SURF 특징 추출
surf = cv2.xfeatures2d.SURF_create()
keypoints1, keypoints2, matches = match_features(gray1, gray2, surf)
img1_warp_surf, M_surf, mask_surf = find_homography_and_warp(img1, img2, keypoints1, keypoints2, matches)
matched_img_surf = draw_matches(img1, keypoints1, img2, keypoints2, matches)

# ORB 특징 추출
orb = cv2.ORB_create()
keypoints1, keypoints2, matches = match_features(gray1, gray2, orb)
img1_warp_orb, M_orb, mask_orb = find_homography_and_warp(img1, img2, keypoints1, keypoints2, matches)
matched_img_orb = draw_matches(img1, keypoints1, img2, keypoints2, matches)

# 결과 출력
plt.figure(figsize=(20, 10))

plt.subplot(3, 2, 1)
plt.title('SIFT Matches', fontsize=16)
plt.imshow(cv2.cvtColor(matched_img_sift, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(3, 2, 2)
plt.title('SIFT Warp', fontsize=16)
plt.imshow(cv2.cvtColor(img1_warp_sift, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(3, 2, 3)
plt.title('SURF Matches', fontsize=16)
plt.imshow(cv2.cvtColor(matched_img_surf, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(3, 2, 4)
plt.title('SURF Warp', fontsize=16)
plt.imshow(cv2.cvtColor(img1_warp_surf, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(3, 2, 5)
plt.title('ORB Matches', fontsize=16)
plt.imshow(cv2.cvtColor(matched_img_orb, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(3, 2, 6)
plt.title('ORB Warp', fontsize=16)
plt.imshow(cv2.cvtColor(img1_warp_orb, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.tight_layout()
plt.savefig('./data/result.png', bbox_inches='tight')
plt.show()