import cv2
import numpy as np
import matplotlib.pyplot as plt

loaded_image = cv2.imread('./data/image_source.png')
cv2.imshow('test', loaded_image)
cv2.waitKey()
B, G, R = cv2.split(loaded_image)




# grey = cv2.imread('./data/Lena.png', 0)
# cv2.imshow('original grey', grey)
# cv2.waitKey()
#
# hist, bins = np.histogram(grey, 256, [0, 255])
# plt.fill(hist)
# plt.xlabel('pixel value')
# plt.show()
#
# grey_eq = cv2.equalizeHist(grey)
# hist, bins = np.histogram(grey_eq, 256, [0, 255])
# plt.fill_between(range(256), hist, 0)
# plt.xlabel('pixel value')
# plt.show()
#
# cv2.imshow('equalized grey', grey_eq)
# cv2.waitKey()
#
#
#
# def plot_histogram_and_equalize(channel, title):
#     # 히스토그램 계산
#     hist, bins = np.histogram(channel.flatten(), 256, [0, 256])
#     # 원본 채널의 히스토그램 표시
#     plt.figure(figsize=(10, 4))
#     plt.subplot(1, 2, 1)
#     plt.fill_between(range(256), hist, 0)
#     plt.title(f'{title} Channel Histogram')
#
#     # 히스토그램 평탄화 수행
#     equalized_channel = cv2.equalizeHist(channel)
#
#     # 평탄화된 채널의 히스토그램 계산 및 표시
#     hist, bins = np.histogram(equalized_channel.flatten(), 256, [0, 256])
#     plt.subplot(1, 2, 2)
#     plt.fill_between(range(256), hist, 0)
#     plt.title(f'Equalized {title} Channel Histogram')
#     plt.show()
#
#     return equalized_channel
#
#
# def process_image_and_plot(image_path, channel_choice):
#     image = cv2.imread(image_path)
#     if image is None:
#         print("Image not found. Please check the path.")
#         return
#
#     # BGR에서 RGB로 변환
#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#
#     if channel_choice.upper() == 'R':
#         channel_index = 0
#     elif channel_choice.upper() == 'G':
#         channel_index = 1
#     elif channel_choice.upper() == 'B':
#         channel_index = 2
#     else:
#         print("Invalid channel choice. Please select 'R', 'G', or 'B'.")
#         return
#
#     # 선택된 채널 추출
#     selected_channel = image_rgb[:, :, channel_index]
#     # 히스토그램 및 평탄화 처리
#     equalized_channel = plot_histogram_and_equalize(selected_channel, channel_choice)
#
#     # 평탄화된 채널로 이미지 업데이트
#     equalized_image = image_rgb.copy()
#     equalized_image[:, :, channel_index] = equalized_channel
#
#     # 원본 이미지 및 처리된 이미지 표시
#     plt.figure(figsize=(10, 5))
#     plt.subplot(1, 2, 1)
#     plt.imshow(image_rgb)
#     plt.title('Original Image')
#
#     plt.subplot(1, 2, 2)
#     plt.imshow(equalized_image)
#     plt.title(f'Image with Equalized {channel_choice} Channel')
#     plt.show()
#
#
# # 사용 예시
# # 'R', 'G', 또는 'B' 중 하나를 선택하여 아래 함수를 호출하세요.
# process_image_and_plot('./data/Lena.png', 'R')
