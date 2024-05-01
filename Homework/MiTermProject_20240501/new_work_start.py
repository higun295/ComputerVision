import cv2
import numpy as np

# 이미지 로드 및 사전 처리
image = cv2.imread('./Data/C_Key_resized.jpg')
new_width = 1200
ratio = new_width / image.shape[1]
new_height = int(image.shape[0] * ratio)

# 이미지 크기 조정
resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

# Bilateral 필터 적용
filtered_image = cv2.bilateralFilter(gray_image, 9, 75, 75)

# 이진화
_, binary_image = cv2.threshold(filtered_image, 80, 255, cv2.THRESH_BINARY_INV)

# 모폴로지 연산
kernel = np.ones((3, 3), np.uint8)
opened_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)

# Sobel 엣지 검출
sobelx = cv2.Sobel(opened_image, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(opened_image, cv2.CV_64F, 0, 1, ksize=5)
sobel = cv2.magnitude(sobelx, sobely)

# 결과 표시
cv2.imshow('Filtered Image', filtered_image)
cv2.imshow('Binary Image', binary_image)
cv2.imshow('Opened Image', opened_image)
cv2.imshow('Sobel Edge', sobel)
cv2.waitKey(0)
cv2.destroyAllWindows()
