import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn import svm

# 이미지 경로
img1_path = './data/20240610_160028943_iOS.jpg'
img2_path = './data/20240610_160030721_iOS.jpg'

# 이미지 불러오기
img1 = cv2.imread(img1_path)
img2 = cv2.imread(img2_path)

# 그레이스케일로 변환
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# SIFT 객체 생성
sift = cv2.SIFT_create()

# 특징점과 디스크립터 추출
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

# 매칭 객체 생성
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = bf.match(des1, des2)

# RANSAC을 사용하여 이상치 제거 및 호모그래피 계산 함수
def find_homography_ransac(kp1, kp2, matches):
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    return H, mask

# 호모그래피 계산 및 시각화
H, mask = find_homography_ransac(kp1, kp2, matches)
matches_mask = mask.ravel().tolist()

# 이미지 차이 계산
height, width = gray2.shape
warped_img1 = cv2.warpPerspective(gray1, H, (width, height))
diff = cv2.absdiff(warped_img1, gray2)

# 차이 이미지에서 임계값 적용
_, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

# 팽창 및 침식으로 노이즈 제거
kernel = np.ones((5,5), np.uint8)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# 윤곽선 검출
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# HoG 특징 추출 및 SVM 분류
def get_hog_features(img):
    # 너무 작은 이미지를 처리하지 않도록 필터링
    if img.shape[0] < 32 or img.shape[1] < 32:
        return np.zeros(36)  # HoG 특징 벡터 크기에 맞는 0 배열 반환
    features = hog(img, pixels_per_cell=(16, 16), cells_per_block=(2, 2), visualize=False)
    return features

# SVM 분류기 학습 (가상의 학습 데이터를 사용하여 예시)
# 실제 프로젝트에서는 눌린/안 눌린 건반의 예시 데이터를 사용하여 학습시켜야 합니다.
svm_clf = svm.SVC(gamma='scale')
# X_train, y_train은 눌린/안 눌린 건반의 HoG 특징과 레이블이 있어야 함
# svm_clf.fit(X_train, y_train)

# 원본 이미지에 눌린 건반 표시
result_img = img2.copy()
for contour in contours:
    if cv2.contourArea(contour) > 100:  # 작은 노이즈 제거
        x, y, w, h = cv2.boundingRect(contour)
        if x < 0 or y < 0 or x + w > gray2.shape[1] or y + h > gray2.shape[0]:
            continue  # 이미지 경계를 벗어나는 경우 제외
        key_roi = gray2[y:y+h, x:x+w]
        hog_features = get_hog_features(key_roi)
        # 눌림 여부 예측 (여기서는 예시로 임의의 결과를 사용)
        # pressed = svm_clf.predict([hog_features])[0]
        pressed = 1  # 가상의 결과, 실제로는 SVM 분류기에서 예측해야 함
        color = (0, 255, 0) if pressed else (0, 0, 255)
        cv2.rectangle(result_img, (x, y), (x+w, y+h), color, 2)

# 결과 시각화
plt.figure(figsize=(20, 10))
plt.subplot(131), plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)), plt.title('Image 1')
plt.subplot(132), plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)), plt.title('Image 2')
plt.subplot(133), plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)), plt.title('Detected Pressed Keys')
plt.show()