Color image capture
Color space conversion
Gamma correction
Histogram Equalization
Image Filtering
Sobel filter for image gradient
Unsharp mask for image sharpening
Discrete Fourier Transform
Frequency-domain Image Filtering
Image Thresholding
Morphological Filter
Edge Detection
Hough Transform
Connected Component Labeling
Region-based growing
Watershed
Image Segmentation using K-means clustering
GrabCut

Corner Detection
	- Harris Corner
	- FAST
	- Good Feature To Track
SIFT (Scale Invariant Feature Transform)

======================================================================

Invariant feature matching
RANSAC
Bag of Visual Word
Optical flow
Panorama stitching
Pinhole camera model vs Lens camera model
Camera Projection Matrix
Geometric Camera Calibration
Radial Distortion
Triangulation
Epipolar Geometry
Essential/Fundamental Matrix
Stereo Rectification
HoG(Histogram of Oriented Gradient)
K-Nearest Neighbor
SVM(Support Vector Machine)
Haar-like feature
Adaboost
 
 
 
 
 
1. 데이터 수집
촬영 장비: 카메라를 사용하여 피아노 건반을 촬영합니다. 카메라는 피아노 건반을 전체적으로 볼 수 있는 위치에 고정합니다.
데이터: 여러 각도에서 다양한 조명 조건에서의 피아노 건반 이미지를 수집합니다.
2. 전처리 및 특징 추출
기술: 불변 특징 매칭 (Invariant feature matching), HoG (Histogram of Oriented Gradients)
설명: 촬영된 이미지에서 건반의 경계를 인식하기 위해 불변 특징 매칭을 사용합니다. HoG를 통해 손가락의 특징을 추출할 수 있습니다.
3. 건반 경계 검출
기술: RANSAC, 기하학적 카메라 보정 (Geometric Camera Calibration)
설명: 건반의 경계를 검출하기 위해 RANSAC을 사용하여 이상치를 제거합니다. 기하학적 카메라 보정을 통해 카메라 왜곡을 보정합니다.
4. 건반 눌림 여부 검출
기술: Optical Flow, K-최근접 이웃 (K-Nearest Neighbor), SVM (Support Vector Machine)
설명: Optical Flow를 사용하여 손가락의 움직임을 추적하고, K-최근접 이웃이나 SVM을 통해 눌림 여부를 분류합니다.
5. 결과 시각화
기술: OpenCV 등을 사용하여 시각화
설명: 눌린 건반을 이미지나 영상에서 실시간으로 표시합니다. OpenCV를 사용하여 결과를 시각화할 수 있습니다.
상세 단계:
영상 캡처 및 전처리:

카메라를 사용하여 피아노 건반을 촬영합니다.
이미지의 조명과 색상을 보정하여 특징 추출에 용이하도록 합니다.
건반 경계 검출:

불변 특징 매칭을 사용하여 건반의 경계를 검출합니다.
RANSAC을 사용하여 건반의 경계 중 이상치를 제거하고, 기하학적 카메라 보정을 통해 정확한 경계를 얻습니다.
건반 눌림 여부 판단:

Optical Flow를 사용하여 손가락의 움직임을 추적합니다.
건반이 눌렸는지 여부를 판단하기 위해 HoG를 사용하여 손가락의 특징을 추출하고, K-최근접 이웃이나 SVM을 사용하여 눌림 여부를 분류합니다.
실시간 결과 표시:

눌린 건반을 실시간으로 표시합니다. 예를 들어, 눌린 건반을 빨간색으로 표시하거나, 텍스트로 "눌림"이라고 표시합니다.
OpenCV를 사용하여 영상에 결과를 시각화합니다.