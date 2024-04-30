# 이미지 크기 조정: 알고리즘의 효율성을 위해 이미지의 해상도를 조정할 수 있습니다.
# 노이즈 제거: 조명 반사나 그림자와 같은 노이즈를 제거하기 위해 이미지 필터링을 적용할 수 있습니다.
# 이진화: 건반과 손가락을 구분하기 위해 이진화 처리를 할 수 있습니다. 이를 위해 이미지의 명암을 기준으로 특정 임곗값을 설정해 건반과 손가락 부분을 명확히 구분할 수 있습니다.
# 색공간 변환: RGB 색공간에서 회색조(Greyscale) 또는 다른 색공간으로 변환하여 데이터를 단순화할 수 있습니다.
# 에지 감지: 손가락이 건반 위에 있는 경계를 찾기 위해 Sobel 필터나 다른 에지 감지 기법을 적용할 수 있습니다.
# 모폴로지 연산: 이진화한 이미지에 대해 모폴로지 연산(예: 팽창, 침식)을 수행하여 손가락과 건반의 경계를 더욱 명확하게 할 수 있습니다.

# 모폴로지 연산 적용: 이진화된 이미지에 모폴로지 연산을 적용하여 손가락과 건반 사이의 구분을 더욱 명확히 할 수 있습니다. 예를 들어, cv2.dilate와 cv2.erode를 사용할 수 있습니다.
# 엣지 감지: 이진화 이미지에서 손가락과 건반의 경계를 찾기 위해 캐니 엣지 디텍터(Canny Edge Detector) 같은 알고리즘을 사용할 수 있습니다.
# 컨투어 검출: cv2.findContours 함수를 사용하여 손가락이나 건반의 경계를 형성하는 연속된 포인트를 찾을 수 있습니다.
# 키스트로크 감지: 검출된 컨투어를 분석하여 눌린 건반을 식별합니다. 이 과정에서 눌린 건반의 위치와 손가락의 위치를 매핑하여 어떤 건반이 눌렸는지를 판단할 수 있습니다.

import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('./Data/C_Key.jpg')
new_width = 1200
ratio = new_width / image.shape[1]
new_height = int(image.shape[0] * ratio)

# 이미지 크기 조정
resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

# 노이즈 제거
blurred_image = cv2.GaussianBlur(resized_image, (0, 0), 1)
# cv2.imshow("blurred_image", blurred_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# 색공간 변환
gray_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2GRAY)
# cv2.imshow("gray_image", gray_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# ==================================================================
# 이진화
# blockSizes = [11, 21, 31]
# Cs = [5, 8, 11]
#
# fig, axes = plt.subplots(3, 3, figsize=(12, 12))
#
# for i, blockSize in enumerate(blockSizes):
#     for j, C in enumerate(Cs):
#         adaptive_thresh = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
#                                                 blockSize, C)
#         ax = axes[i, j]
#         ax.imshow(adaptive_thresh, cmap='gray')
#         ax.set_title(f'BlockSize: {blockSize}, C: {C}')
#         ax.axis('off')
#
# plt.show()
#
adaptive_thresh = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
# cv2.imwrite('./Data/C_Key_adaptive_thresh.jpg', adaptive_thresh)
# cv2.imshow("adaptive_thresh", adaptive_thresh)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# ==================================================================
#
#
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


# 윤곽 검출
_, contours, _ = cv2.findContours(adaptive_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 건반 윤곽 추출
key_contours = []
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    if w > 50 and h > 50:
        key_contours.append(contour)

# 건반 윤곽을 사용하여 마스크 생성
mask = np.zeros_like(gray_image)
cv2.drawContours(mask, key_contours, -1, 255, thickness=cv2.FILLED)

# 마스크를 적용하여 건반 영역 추출
key_region = cv2.bitwise_and(image, image, mask=mask)

# 결과 표시
cv2.imshow('Piano Keys', key_region)
cv2.waitKey(0)
cv2.destroyAllWindows()


# _, contours, hierarchy = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
# # 윤곽선을 이미지에 그립니다. -1은 모든 윤곽을 그린다는 의미입니다.
# # 녹색으로 윤곽선을 그리고 두께는 2로 설정합니다.
# contour_image = cv2.drawContours(resized_image.copy(), contours, -1, (0, 255, 0), 2)
#
# # 결과 이미지를 화면에 표시합니다
# cv2.imshow('Contours', contour_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()