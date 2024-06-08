import cv2
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import exposure

# 이미지 불러오기
image_path = './data/piano_1.jpg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Gaussian Blur 적용하여 반짝임 감소
blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

# CLAHE 적용하여 대비 조정
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
clahe_image = clahe.apply(blurred_image)

# HoG 특징 추출
hog_features, hog_image = hog(clahe_image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True, feature_vector=True)

# HoG 이미지 시각적으로 강화
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

# 원본 이미지 시각화
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

# Gaussian Blur 이미지 시각화
plt.subplot(1, 3, 2)
plt.imshow(blurred_image, cmap='gray')
plt.title('Gaussian Blurred Image')
plt.axis('off')

# CLAHE 이미지 시각화
plt.subplot(1, 3, 3)
plt.imshow(clahe_image, cmap='gray')
plt.title('CLAHE Image')
plt.axis('off')

# HoG 특징 이미지 시각화
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(hog_image_rescaled, cmap='gray')
plt.title('HoG Image')
plt.axis('off')

plt.show()
