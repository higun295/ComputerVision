import cv2
import numpy as np
import matplotlib.pyplot as plt

# 이미지 경로
img1_path = './data/1_1.jpg'
img2_path = './data/1_3.jpg'

# 이미지 불러오기
img1 = cv2.imread(img1_path)
img2 = cv2.imread(img2_path)

# 이미지가 제대로 로드되었는지 확인
if img1 is None:
    raise FileNotFoundError(f"Image at path '{img1_path}' could not be loaded.")
if img2 is None:
    raise FileNotFoundError(f"Image at path '{img2_path}' could not be loaded.")

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

# 특징점 매칭 객체 생성
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = bf.match(des1, des2)

# 매칭된 점 집합 생성
pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 2)

# RANSAC을 사용하여 Fundamental Matrix 계산
F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)

# RANSAC 결과로 필터링된 매칭 결과
pts1 = pts1[mask.ravel() == 1]
pts2 = pts2[mask.ravel() == 1]

print("Fundamental Matrix:")
print(F)

# 카메라의 내부 파라미터 설정
focal_length = 13  # focal length in mm
image_width = img1.shape[1]
image_height = img1.shape[0]

# Convert focal length to pixels (assuming a sensor width of 6.4mm for this example)
sensor_width_mm = 6.4
fx = (focal_length / sensor_width_mm) * image_width
fy = fx  # assuming square pixels

# Principal point
cx = image_width / 2
cy = image_height / 2

# Camera intrinsic matrix K
K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0, 0, 1]])

print("Camera Matrix K:")
print(K)

# Essential Matrix 계산
E = K.T @ F @ K

print("Essential Matrix:")
print(E)

# Essential Matrix 분해하여 R (회전 행렬)과 T (변환 벡터) 계산
_, R, T, _ = cv2.recoverPose(E, pts1, pts2, K)

print("Rotation Matrix:")
print(R)
print("Translation Vector:")
print(T)

# 카메라의 각도 계산 (Euler Angles)
def rotation_matrix_to_euler_angles(R):
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0

    return np.degrees(np.array([x, y, z]))

euler_angles = rotation_matrix_to_euler_angles(R)
print("Euler Angles (degrees):")
print(euler_angles)

# Epipolar lines 계산 및 시각화
def draw_epipolar_lines(img1, img2, pts1, pts2, F, num_lines=100):
    # 두 이미지의 epilines 계산
    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F).reshape(-1, 3)
    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F).reshape(-1, 3)

    # 이미지 복사본 생성
    img1_color = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2_color = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    # 선택된 에피폴라 라인만 그리기
    selected_indices = np.random.choice(len(pts1), num_lines, replace=False)

    for i in selected_indices:
        r = lines1[i]
        pt1 = pts1[i]
        pt2 = pts2[i]
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [img1.shape[1], -(r[2] + r[0] * img1.shape[1]) / r[1]])
        img1_color = cv2.line(img1_color, (x0, y0), (x1, y1), color, 1)
        img1_color = cv2.circle(img1_color, tuple(map(int, pt1)), 5, color, -1)

    for i in selected_indices:
        r = lines2[i]
        pt1 = pts1[i]
        pt2 = pts2[i]
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [img2.shape[1], -(r[2] + r[0] * img2.shape[1]) / r[1]])
        img2_color = cv2.line(img2_color, (x0, y0), (x1, y1), color, 1)
        img2_color = cv2.circle(img2_color, tuple(map(int, pt2)), 5, color, -1)

    return img1_color, img2_color

img1_epi, img2_epi = draw_epipolar_lines(gray1, gray2, pts1, pts2, F, num_lines=400)

# Epipolar lines 시각화
plt.figure(figsize=(20, 10))
plt.subplot(121), plt.imshow(cv2.cvtColor(img1_epi, cv2.COLOR_BGR2RGB)), plt.title('Image 1 with Epipolar Lines')
plt.subplot(122), plt.imshow(cv2.cvtColor(img2_epi, cv2.COLOR_BGR2RGB)), plt.title('Image 2 with Epipolar Lines')
plt.show()
