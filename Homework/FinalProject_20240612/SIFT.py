import cv2
import numpy as np

# 이미지 불러오기
img0 = cv2.imread('./data/20240610_160028943_iOS.jpg', cv2.IMREAD_COLOR)
img1 = cv2.imread('./data/20240610_160030721_iOS.jpg', cv2.IMREAD_COLOR)

# 두 이미지를 절반 이하로 리사이즈 (0.4 배율 적용)
scale_factor = 0.2
img0 = cv2.resize(img0, (int(img0.shape[1] * scale_factor), int(img0.shape[0] * scale_factor)))
img1 = cv2.resize(img1, (int(img1.shape[1] * scale_factor), int(img1.shape[0] * scale_factor)))

# 이미지 크기 조정 및 패딩 추가
img1 = np.pad(img1, ((64,) * 2, (64,) * 2, (0,) * 2), 'constant', constant_values=0)


# 두 이미지의 높이와 너비를 동일하게 맞추기 위해 패딩 추가
def pad_to_same_size(img1, img2):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    if h1 != h2:
        if h1 > h2:
            pad_top_bottom = (h1 - h2) // 2
            img2 = np.pad(img2, ((pad_top_bottom, h1 - h2 - pad_top_bottom), (0, 0), (0, 0)), 'constant',
                          constant_values=0)
        else:
            pad_top_bottom = (h2 - h1) // 2
            img1 = np.pad(img1, ((pad_top_bottom, h2 - h1 - pad_top_bottom), (0, 0), (0, 0)), 'constant',
                          constant_values=0)

    if w1 != w2:
        if w1 > w2:
            pad_left_right = (w1 - w2) // 2
            img2 = np.pad(img2, ((0, 0), (pad_left_right, w1 - w2 - pad_left_right), (0, 0)), 'constant',
                          constant_values=0)
        else:
            pad_left_right = (w2 - w1) // 2
            img1 = np.pad(img1, ((0, 0), (pad_left_right, w2 - w1 - pad_left_right), (0, 0)), 'constant',
                          constant_values=0)

    return img1, img2


img0, img1 = pad_to_same_size(img0, img1)

# 이미지 리스트
imgs_list = [img0, img1]

# SIFT 특징점 검출기 생성
detector = cv2.SIFT_create(50)

# 특징점 검출 및 그리기
for i in range(len(imgs_list)):
    keypoints, descriptors = detector.detectAndCompute(imgs_list[i], None)
    imgs_list[i] = cv2.drawKeypoints(imgs_list[i], keypoints, None, (0, 255, 0),
                                     flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# 이미지 연결 및 시각화
concatenated_imgs = np.hstack(imgs_list)

cv2.imshow('SIFT keypoints', concatenated_imgs)
cv2.waitKey()
cv2.destroyAllWindows()