import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import exposure
from sklearn.metrics.pairwise import cosine_similarity

# 이미지 경로
img1_path = './data/1_1.jpg'
img2_path = './data/1_2.jpg'

# 이미지 불러오기
img1 = cv2.imread(img1_path)
img2 = cv2.imread(img2_path)

# 이미지가 제대로 로드되었는지 확인
if img1 is None:
    raise FileNotFoundError(f"Image at path '{img1_path}' could not be loaded.")
if img2 is None:
    raise FileNotFoundError(f"Image at path '{img2_path}' could not be loaded.")

# 동일한 크기로 이미지 조정
resize_dim = (400, 400)  # 예시로 400x400으로 조정
img1 = cv2.resize(img1, resize_dim)
img2 = cv2.resize(img2, resize_dim)

# 그레이스케일로 변환
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# HoG 특징 추출
def extract_hog_features(img):
    features, hog_image = hog(img, pixels_per_cell=(16, 16), cells_per_block=(2, 2), visualize=True)
    return features, hog_image

features1, hog_image1 = extract_hog_features(gray1)
features2, hog_image2 = extract_hog_features(gray2)

# HoG 이미지 시각화
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10), sharex=True, sharey=True)

# 원본 이미지와 HoG 이미지 비교
ax1.axis('off')
ax1.imshow(gray1, cmap=plt.cm.gray)
ax1.set_title('Input Image 1')

# HoG 이미지
hog_image_rescaled1 = exposure.rescale_intensity(hog_image1, in_range=(0, 10))
ax2.axis('off')
ax2.imshow(hog_image_rescaled1, cmap=plt.cm.gray)
ax2.set_title('HoG Features 1')

plt.show()

fig, (ax3, ax4) = plt.subplots(1, 2, figsize=(20, 10), sharex=True, sharey=True)

# 원본 이미지와 HoG 이미지 비교
ax3.axis('off')
ax3.imshow(gray2, cmap=plt.cm.gray)
ax3.set_title('Input Image 2')

# HoG 이미지
hog_image_rescaled2 = exposure.rescale_intensity(hog_image2, in_range=(0, 10))
ax4.axis('off')
ax4.imshow(hog_image_rescaled2, cmap=plt.cm.gray)
ax4.set_title('HoG Features 2')

plt.show()

# 히스토그램 비교
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.hist(features1, bins=50, alpha=0.5, label='Image 1 HoG Features')
ax.hist(features2, bins=50, alpha=0.5, label='Image 2 HoG Features')
ax.legend(loc='upper right')
ax.set_title('HoG Feature Histograms')
plt.show()

# 코사인 유사도 계산
cos_sim = cosine_similarity([features1], [features2])[0][0]

print(f'Cosine Similarity between Image 1 and Image 2: {cos_sim:.4f}')

# 히트맵 시각화
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
cax = ax.matshow(cosine_similarity([features1], [features2]), cmap='viridis')
plt.colorbar(cax)
ax.set_xticks([0])
ax.set_yticks([0])
ax.set_xticklabels(['Image 2'])
ax.set_yticklabels(['Image 1'])
ax.set_title('Cosine Similarity Heatmap')
plt.show()
