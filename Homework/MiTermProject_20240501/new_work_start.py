import cv2
import numpy as np

# 이미지 로드 및 그레이스케일 변환
image = cv2.imread('./Data/C_Key_resized.jpg')
new_width = 1200
ratio = new_width / image.shape[1]
new_height = int(image.shape[0] * ratio)

# 이미지 크기 조정
resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

# 이진화를 통해 건반과 손가락을 더 잘 구분할 수 있게 처리
_, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

# 윤곽 검출
_, contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 각 윤곽에 대해 반복
for contour in contours:
    # 윤곽의 경계 상자를 구합니다
    x, y, w, h = cv2.boundingRect(contour)

    # 간단한 건반 크기 필터 (건반 크기 추정을 통해 설정)
    if 20 < w < 600 and 100 < h < 600:
        # 경계 상자를 이미지에 그립니다
        cv2.rectangle(resized_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# 결과 이미지 표시
cv2.imshow('Detected Keys', resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
