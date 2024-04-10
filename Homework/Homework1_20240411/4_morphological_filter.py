# 1. 사용자의 입력을 받아서 Otsu, Median 중 선택
# 2. 선택한 이진화 기법 적용하여 결과 출력
# 3. Erosion, Dilation, Opening, Closing에 대한 선택과 횟수를 입력 받음
# 4. 해당 결과 출력

import cv2
import numpy as np
from matplotlib import pyplot as plt

image = cv2.imread('./data/Lena.png', 0)

print("이진화 방법을 선택하세요 : ")
print("1 : Otsu's Binarization")
print("2 : Adaptive Threshold")
binary_user_input = input("선택 (1, 2) : ")

# 1. 사용자의 입력을 받아서 Otsu, Median 중 선택
if binary_user_input == '1':
    ret, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    title = "Otsu's Binarization"
elif binary_user_input == '2':
    thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    title = "Adaptive Threshold"

# 2. 선택한 이진화 기법 적용하여 결과 출력
plt.imshow(thresh, cmap='gray')
plt.title(title)
plt.axis('off')
plt.show()

# 3. Erosion, Dilation, Opening, Closing에 대한 선택과 횟수를 입력 받음
print("모폴로지 연산을 선택하세요 : ")
print("1 : Erosion")
print("2 : Dilation")
print("3 : Opening")
print("4 : Closing")
morph_user_input = input("선택 (1, 2, 3, 4) : ")
iterations = int(input("적용 횟수를 입력하세요 : "))

kernel = np.ones((5, 5), np.uint8)

if morph_user_input == '1':
    result = cv2.erode(thresh, kernel, iterations=iterations)
    title = "Erosion"
elif morph_user_input == '2':
    result = cv2.dilate(thresh, kernel, iterations=iterations)
    title = "Dilation"
elif morph_user_input == '3':
    result = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=iterations)
    title = "Opening"
elif morph_user_input == "4":
    result = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    title = "Closing"

# 4. 해당 결과 출력
plt.imshow(result, cmap='gray')
plt.title(title)
plt.axis('off')
plt.show()