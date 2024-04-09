# 1. 사용자로부터 R, G, B 입력 받기
# 2. 입력 받은 채널에 대한 히스토그램 그리기
# 3. 히스토그램에 대해서 평탄화 작업 진행
# 4. 평탄화 이후 영상 출력
# 5. 출력한 영상에 대해 HSV 컬러 스페이스로 변경
# 6. V 채널에 대한 평탄화 진행
# 7. 평탄화 이후 영상 출력

import cv2
import numpy as np
from matplotlib import pyplot as plt

original_image = cv2.imread('./data/image_source.png')
cv2.imshow('ORIGINAL IMAGE', original_image)

# 1. 사용자로부터 R, G, B 입력 받기
key = cv2.waitKey(0) & 0xFF


def equalize_histogram(channel_index, channel_data):
    # 3. 히스토그램에 대해서 평탄화 작업 진행
    equalized_channel = cv2.equalizeHist(channel_data)
    equalized_image = original_image.copy()
    equalized_image[:, :, channel_index] = equalized_channel

    # 4. 평탄화 이후 영상 출력
    cv2.imshow('EQUALIZED IMAGE', equalized_image)

    # 5. 출력한 영상에 대해 HSV 컬러 스페이스로 변경
    hsv = cv2.cvtColor(equalized_image, cv2.COLOR_BGR2HSV)

    # 6. V 채널에 대한 평탄화 진행
    hsv[..., 2] = cv2.equalizeHist(hsv[..., 2])
    final_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # 7. 평탄화 이후 영상 출력
    cv2.imshow('FINAL IMAGE', final_image)


if key in [ord('r'), ord('g'), ord('b')]:
    # OpenCV는 BGR 순서로 채널을 관리함
    channel_index = {'b': 0, 'g': 1, 'r': 2}[chr(key)]
    channel_data = original_image[:, :, channel_index]

    hist, bins = np.histogram(channel_data, 256, [0, 256])

    # 2. 입력받은 채널에 대한 히스토그램 그리기
    plt.fill_between(range(256), hist, 0)
    plt.title(f'Histogram of {chr(key).upper()} Channel')
    plt.xlabel('pixel value')
    plt.ylabel('frequency')
    plt.show()

    equalize_histogram(channel_index, channel_data)

cv2.waitKey()
cv2.destroyAllWindows()