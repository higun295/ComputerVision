# CreaterStitcher 함수를 이용하여 4개의 영상 셋에 대해서 파노라마 이미지를 만드는 방법을 구현하시오.

import cv2
import numpy as np
import matplotlib.pyplot as plt

# 이미지 경로 설정
img1_path = './data/boat1.jpg'
img2_path = './data/boat2.jpg'
img3_path = './data/boat3.jpg'
img4_path = './data/boat4.jpg'

# 이미지 읽기
img1 = cv2.imread(img1_path)
img2 = cv2.imread(img2_path)
img3 = cv2.imread(img3_path)
img4 = cv2.imread(img4_path)

# Stitcher 객체 생성
stitcher = cv2.createStitcher() if int(cv2.__version__.split('.')[0]) < 4 else cv2.Stitcher_create()

# 이미지 리스트
images = [img1, img2, img3, img4]

# 파노라마 생성
status, pano = stitcher.stitch(images)

# 결과 출력
plt.figure(figsize=(30, 15))
plt.imshow(cv2.cvtColor(pano, cv2.COLOR_BGR2RGB))
plt.title('Panorama Image', fontsize=25)
plt.axis('off')
plt.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.05, wspace=0.1, hspace=0.1)
plt.show()
