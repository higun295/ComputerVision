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

# 그레이스케일로 변환
gray_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2GRAY)

# Bilateral 필터 적용
filtered_image = cv2.bilateralFilter(gray_image, 9, 75, 75)

# 이진화
adaptive_thresh = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

# 모폴로지 연산 - 세밀한 조정
kernel = np.ones((2, 2), np.uint8)  # 작은 커널 사용
opened = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_OPEN, kernel, iterations=1)  # 반복 횟수 감소

# Sobel 엣지 검출
sobelx = cv2.Sobel(opened, cv2.CV_64F, 1, 0, ksize=3)  # x 방향 엣지
sobely = cv2.Sobel(opened, cv2.CV_64F, 0, 1, ksize=3)  # y 방향 엣지
sobel = cv2.magnitude(sobelx, sobely)  # 엣지 강도 계산

# 윤곽 검출
_, contours, _ = cv2.findContours(sobel.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 각 윤곽에 대해 경계 상자 그리기
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    if h / w > 2:
        cv2.rectangle(resized_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# 결과 표시
cv2.imshow('Detected Piano Keys', resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# # 결과 표시
# cv2.imshow('Opened', opened)
# cv2.imshow('Sobel Edge', sobel)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# # 모폴로지 연산 - Opening과 Closing 적용
# kernel = np.ones((2, 2), np.uint8)
# opened = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_OPEN, kernel)
# closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
#
# # 엣지 검출
# edges = cv2.Canny(closed, 50, 150)
# cv2.imshow('Opened', opened)
# cv2.imshow('Closed', closed)
# cv2.imshow('Edges', edges)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

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
