데이터 수집

피아노의 여러 부분을 다양한 각도에서 촬영한 이미지 세트를 수집합니다.
예를 들어, 피아노의 건반, 몸체, 뚜껑 등을 포함한 이미지를 촬영합니다.
특징점 추출 및 매칭

SIFT, ORB 등의 알고리즘을 사용하여 피아노 이미지의 특징점을 추출하고 매칭합니다.
RANSAC을 통한 이상치 제거

특징점 매칭 과정에서의 이상치를 제거하여 정확한 매칭을 보장합니다.
호모그래피 추정

특징점 매칭 결과를 바탕으로 두 이미지 간의 호모그래피를 추정합니다.
이를 통해 이미지를 정렬하고 동일한 평면으로 투영합니다.
왜곡 보정 및 블렌딩

이미지의 왜곡을 보정하여 자연스러운 정합을 구현합니다.
Multi-band blending을 사용하여 여러 이미지를 자연스럽게 연결합니다.
결과 시각화

중간 과정과 최종 파노라마 이미지를 시각화하여 결과를 평가합니다.
기대 결과
이 프로젝트를 통해 피아노의 전체 모습을 하나의 파노라마 이미지로 합성하는 과정을 이해하고, 다양한 컴퓨터 비전 알고리즘을 실습할 수 있습니다. 또한, 결과물을 통해 실제 응용 가능성을 확인할 수 있습니다.




피아노의 이미지를 촬영해서, SIFT, ORB 등의 알고리즘을 사용하여 이미지의 특징점을 추출하려고 했는데 모양과 색이 비슷한 부분이 많아서 제대로 탐지하지 못함. 