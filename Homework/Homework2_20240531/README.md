<h1>Homework#2</h1>
<h2>1. Feature Detection</h2>
Stitching.zip에서 4장의 영상(boat1, budapest1, newspaper1,sl)을 선택한 후에 Canny Edge와 Harris Corner를 검출해서 결과를 출력하는 코드를 작성하시오.

<h2>2. Matching</h2>
Stitching.zip에서 각 영상셋(boat, budapest, newspaper, s1~s2)에서 두 장을 선택하고 각 영상에서 각각 SIFT, SURF, ORB를 추출한 후에 매칭 및 RANSAC을 통해서 두 장의 영상 간의 homography를 계산하고, 이를 통해 한 장의 영상을 다른 한 장의 영상으로 warping하는 코드를 작성하시오.

<h2>3. Panorama</h2>
CreaterStitcher 함수를 이용하여 4개의 영상 셋에 대해서 파노라마 이미지를 만드는 방법을 구현하시오.

<h2>4. Optical Flow</h2>
Stitching.zip에서 dog_a, dog_b 두 사진을 이용해서 Good feature to Track을 추출하고 Pyramid Lucas-Kanade 알고리즘을 적용해서 Optical Flow를 구하는 코드를 작성하시오.