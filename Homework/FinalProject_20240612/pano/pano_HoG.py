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

# Optical Flow 분석 (Farneback 방법)
flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)

# Optical Flow 결과 시각화
hsv = np.zeros_like(img1)
hsv[..., 1] = 255

# Optical Flow의 방향을 각도로 변환
mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
hsv[..., 0] = ang * 180 / np.pi / 2
hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
optical_flow_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

# 결과 시각화
plt.figure(figsize=(20, 10))
plt.subplot(121), plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)), plt.title('Image 1')
plt.subplot(122), plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)), plt.title('Image 2')
plt.show()

plt.figure(figsize=(20, 10))
plt.imshow(optical_flow_img)
plt.title('Optical Flow')
plt.axis('off')
plt.show()
