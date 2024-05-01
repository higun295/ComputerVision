import cv2
import numpy as np

# 이미지 로드
image = cv2.imread('./Data/C_Key_resized.jpg')
new_width = 1200
ratio = new_width / image.shape[1]
new_height = int(image.shape[0] * ratio)

# 이미지 크기 조정
resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

# 노이즈 제거
blurred_image = cv2.GaussianBlur(resized_image, (0, 0), 1)
cv2.imshow("blurred_image", blurred_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 그레이스케일로 변환
gray_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray_image', gray_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 이진화
adaptive_thresh = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
cv2.imshow('Binary Image', adaptive_thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('./Data/find_keyboard_adaptive_thresh.jpg', adaptive_thresh)

# kernel = np.ones((3, 3), np.uint8)
# # 열림 연산 적용
# opening = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_OPEN, kernel, iterations=1)
# # 닫힘 연산 적용
# closing = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
#
# edges = cv2.Canny(opening, 110, 250)
#
# # 엣지 검출 결과를 화면에 표시합니다
# cv2.imshow('Edges', edges)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#
# # 노이즈 제거를 위한 모폴로지 연산
# kernel = np.ones((3, 3), np.uint8)
# binary = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_OPEN, kernel)
# cv2.imshow('Morphology Image', binary)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# # 윤곽 검출
# _, contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cv2.drawContours(resized_image, contours, -1, (0, 255, 0), 3)
# cv2.imshow('Contours on Image', resized_image)  # 윤곽이 그려진 이미지 표시
# cv2.waitKey(0)
#
# # 건반 식별 및 표시
# for contour in contours:
#     x, y, w, h = cv2.boundingRect(contour)
#     # 건반 크기 및 비율 필터링
#     # if h > 10 and w > 10 and h/w > 3:
#     cv2.rectangle(resized_image, (x, y), (x+w, y+h), (0, 255, 0), 2)

# # 결과 표시
# cv2.imshow('Detected Piano Keys', resized_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
