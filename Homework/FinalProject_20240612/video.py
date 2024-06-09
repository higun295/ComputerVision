import cv2
import numpy as np
import tensorflow as tf

# 동영상 파일 경로
video_path = './data/piano_2.mp4'

# 학습된 모델 로드
model = tf.keras.models.load_model('./data/note_model.h5')

# 건반 위치에 따른 음 이름 매핑
key_to_note = {
    0: "C",
    1: "C#",
    2: "D",
    3: "D#",
    4: "E",
    5: "F",
    6: "F#",
    7: "G",
    8: "G#",
    9: "A",
    10: "A#",
    11: "B",
    12: "C",
    # 필요한 만큼 추가
}

# 동영상 캡처 객체 생성
cap = cv2.VideoCapture(video_path)

# SIFT 특징점 검출기 생성
sift = cv2.SIFT_create()

# 첫 번째 프레임 읽기
ret, prev_frame = cap.read()
if not ret:
    print("Failed to read the video.")
    cap.release()
    exit()

# 첫 번째 프레임을 그레이스케일로 변환
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# 펜의 색상 범위 설정 (여기서는 예시로 파란색 펜을 사용한다고 가정)
lower_pen_color = np.array([100, 150, 0])
upper_pen_color = np.array([140, 255, 255])

while True:
    # 다음 프레임 읽기
    ret, curr_frame = cap.read()
    if not ret:
        break

    # 현재 프레임을 그레이스케일로 변환
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

    # 펜의 색상 영역을 마스크로 설정 (BGR to HSV 변환 후 마스킹)
    hsv_frame = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2HSV)
    pen_mask = cv2.inRange(hsv_frame, lower_pen_color, upper_pen_color)

    # 펜을 제외한 영역에서만 변화 검출
    masked_prev_gray = cv2.bitwise_and(prev_gray, prev_gray, mask=cv2.bitwise_not(pen_mask))
    masked_curr_gray = cv2.bitwise_and(curr_gray, curr_gray, mask=cv2.bitwise_not(pen_mask))

    # SIFT 특징점 및 디스크립터 검출
    keypoints1, descriptors1 = sift.detectAndCompute(masked_prev_gray, None)
    keypoints2, descriptors2 = sift.detectAndCompute(masked_curr_gray, None)

    # FLANN 기반 매칭 객체 생성
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # 매칭
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    # 좋은 매칭점만 선택 (Lowe's ratio test)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # RANSAC을 사용하여 호모그래피 행렬 계산
    if len(good_matches) > 4:
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        height, width = prev_gray.shape
        curr_aligned = cv2.warpPerspective(curr_gray, H, (width, height))
    else:
        curr_aligned = curr_gray

    # 차이 이미지 계산
    diff_image = cv2.absdiff(prev_gray, curr_aligned)
    _, thresh = cv2.threshold(diff_image, 30, 255, cv2.THRESH_BINARY)

    # 펜 영역을 마스크로 제거한 후 변화 검출
    masked_thresh = cv2.bitwise_and(thresh, thresh, mask=cv2.bitwise_not(pen_mask))

    # 윤곽선 검출
    contours, _ = cv2.findContours(masked_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 윤곽선에 초록색 박스 그리기
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # 노이즈 제거를 위한 최소 면적 필터링
            x, y, w, h = cv2.boundingRect(contour)
            roi = curr_frame[y:y+h, x:x+w]

            # 이미지 전처리
            img = cv2.resize(roi, (150, 150))
            img = img / 255.0
            img = np.expand_dims(img, axis=0)

            # 예측
            prediction = model.predict(img)
            if prediction > 0.5:
                note_index = np.argmax(prediction)
                note = key_to_note.get(note_index, "Unknown")
                cv2.putText(curr_frame, f'{note}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.rectangle(curr_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            break

    # 결과 시각화
    cv2.imshow('Piano Key Press Detection', curr_frame)

    # ESC 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == 27:
        break

    # 현재 프레임을 이전 프레임으로 업데이트
    prev_gray = curr_gray.copy()

# 자원 해제
cap.release()
cv2.destroyAllWindows()
