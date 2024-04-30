import cv2
import numpy as np

# 이미지를 로드합니다.
image = cv2.imread('./Data/C_Key_resized.jpg')

# 이미지 사이즈 해상도에 맞게 조정 필요
new_width = 1200
ratio = new_width / image.shape[1]
new_height = int(image.shape[0] * ratio)

# 이미지 크기 조정
resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

# HSV 색공간으로 변환합니다.
hsv_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2HSV)

# HSV 채널 분리합니다.
h, s, v = cv2.split(hsv_image)

# 밝기 채널을 조정합니다.
# 이진화를 사용하여 그림자가 예상되는 영역을 분리할 수 있습니다.
_, low_sat = cv2.threshold(s, 40, 255, cv2.THRESH_BINARY_INV)
_, low_val = cv2.threshold(v, 50, 255, cv2.THRESH_BINARY)

shadow_mask = cv2.bitwise_and(low_val, low_val, mask=low_sat)

# 그림자 영역의 밝기를 증가시킵니다.
v_adjusted = cv2.add(v, shadow_mask)

# HSV 이미지 재조합 및 BGR로 변환
final_hsv = cv2.merge([h, s, v_adjusted])
result_image = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

# 결과를 표시합니다.
# cv2.imshow('Original Image', resized_image)
# cv2.imshow('Shadow Mask', shadow_mask)
cv2.imshow('Result Image', result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()