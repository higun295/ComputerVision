# boat, budapest, newspaper, s1~s2 에서 두 장을 선택하고
# 각 영상에서 각각 SIFT, SURF, ORB를 추출한 후에 매칭 및 RANSAC을 통해서 두 장의 영상 간의 homography를 계산
# 이를 통해 한 장의 영상을 다른 한 장의 영상으로 warping하는 코드 작성

# import cv2
# import matplotlib.pyplot as plt
# import numpy as np
#
# # 이미지 경로
# img1_path = './data/boat1.jpg'
# img2_path = './data/boat2.jpg'
#
# # 이미지 읽기
# img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
# img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
#
# def match_and_warp(img1, img2, detector, title):
#     # 특징점 및 디스크립터 추출
#     keypoints1, descriptors1 = detector.detectAndCompute(img1, None)
#     keypoints2, descriptors2 = detector.detectAndCompute(img2, None)
#
#     # BFMatcher 생성
#     bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
#
#     # 디스크립터 매칭
#     matches = bf.match(descriptors1, descriptors2)
#
#     # 매칭 결과 정렬 (거리가 가까운 순서로)
#     matches = sorted(matches, key=lambda x: x.distance)
#
#     # 매칭 결과 이미지 그리기
#     img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
#
#     plt.figure(figsize=(30, 15))
#     plt.imshow(img_matches)
#     plt.title(f'{title} Feature Matching', fontsize=25)
#     plt.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.05, wspace=0.1, hspace=0.1)
#     plt.show()
#
#     # RANSAC을 사용하여 Homography 계산
#     src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
#     dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 2)
#
#     H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
#     matchesMask = mask.ravel().tolist()
#
#     # Homography 결과를 사용하여 이미지를 변환 (warping)
#     height, width = img2.shape
#     warped_img1 = cv2.warpPerspective(img1, H, (width, height))
#
#     # 원본 이미지와 변환된 이미지 함께 출력
#     plt.figure(figsize=(20, 10))
#     plt.subplot(1, 3, 1)
#     plt.imshow(img1, cmap='gray')
#     plt.title('Original Image 1')
#
#     plt.subplot(1, 3, 2)
#     plt.imshow(img2, cmap='gray')
#     plt.title('Original Image 2')
#
#     plt.subplot(1, 3, 3)
#     plt.imshow(warped_img1, cmap='gray')
#     plt.title('Warped Image 1')
#
#     plt.show()
#
#     # 매칭 결과를 마스크를 사용하여 시각화
#     draw_params = dict(matchColor=(0, 255, 0),  # 매칭점의 색상
#                        singlePointColor=None,
#                        matchesMask=matchesMask,  # inliers 매칭만 그리기
#                        flags=2)
#
#     img_matches_ransac = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches, None, **draw_params)
#
#     plt.figure(figsize=(30, 15))
#     plt.imshow(img_matches_ransac)
#     plt.title(f'{title} Feature Matching with RANSAC', fontsize=25)
#     plt.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.05, wspace=0.1, hspace=0.1)
#     plt.show()
#
# # SIFT
# sift = cv2.SIFT_create()
# match_and_warp(img1, img2, sift, "SIFT")
#
# # SURF (SURF는 특허가 있기 때문에 일부 OpenCV 배포판에 포함되지 않을 수 있음)
# surf = cv2.xfeatures2d.SURF_create()
# match_and_warp(img1, img2, surf, "SURF")
#
# # ORB
# orb = cv2.ORB_create()
# match_and_warp(img1, img2, orb, "ORB")

# import cv2
# import matplotlib.pyplot as plt
#
# # 이미지 경로
# img1_path = './data/boat1.jpg'
# img2_path = './data/boat2.jpg'
#
# # 이미지 읽기
# img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
# img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
#
# # SURF 생성 (hessianThreshold 값을 조정하여 검출할 특징점의 개수를 조절할 수 있음)
# surf = cv2.xfeatures2d.SURF_create(10000)
#
# # SURF 특징점 및 디스크립터 추출
# keypoints1, descriptors1 = surf.detectAndCompute(img1, None)
# keypoints2, descriptors2 = surf.detectAndCompute(img2, None)
#
# # 특징점을 이미지에 그리기
# img1_surf = cv2.drawKeypoints(img1, keypoints1, None, (255, 0, 0), 4)
# img2_surf = cv2.drawKeypoints(img2, keypoints2, None, (255, 0, 0), 4)
#
# # 결과 이미지 출력
# plt.figure(figsize=(20, 10))
# plt.subplot(1, 2, 1)
# plt.imshow(img1_surf)
# plt.title('SURF Keypoints in Image 1')
#
# plt.subplot(1, 2, 2)
# plt.imshow(img2_surf)
# plt.title('SURF Keypoints in Image 2')
#
# plt.show()
