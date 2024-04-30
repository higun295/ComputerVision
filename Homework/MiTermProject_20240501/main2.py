import cv2
import numpy as np

# 이미지 로드 및 그레이스케일 변환
image = cv2.imread('./Data/C_Key_resized.jpg')

# 이미지 사이즈 해상도에 맞게 조정 필요
new_width = 1200
ratio = new_width / image.shape[1]
new_height = int(image.shape[0] * ratio)

# 이미지 크기 조정
resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

# 이진화 (이미 이진화 된 이미지가 있다면 이 단계는 생략)
_, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

# 윤곽 검출
_, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 건반 윤곽 추출
key_contours = []
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    if w > 80 and h > 80:  # 건반의 특성에 맞게 조정
        key_contours.append(contour)

# 건반 윤곽을 사용하여 마스크 생성
mask = np.zeros_like(gray)
cv2.drawContours(mask, key_contours, -1, (255), thickness=cv2.FILLED)

# 마스크를 적용하여 건반 영역 추출
key_region = cv2.bitwise_and(resized_image, resized_image, mask=mask)

# 결과 표시
cv2.imshow('Piano Keys', key_region)
cv2.waitKey(0)
cv2.destroyAllWindows()
