# 얼굴 인식하여 캐릭터 씌우기

# Face Detection (얼굴의 특징을 파악하여 인식) vs Face Recognition (누구의 얼굴인지 알아낼때 사용)

# 구글 검색 : mediapipe -> https://developers.google.com/mediapipe

# 패키지 설치 pip install mediapipe

import cv2
import mediapipe as mp

def overlay(image, x, y, w, h, overlay_image): # 대상이미지(3채널), x, y 좌표, widht, height, 덮어씌울 이미지(4채널)
    alpha = overlay_image[:, :, 3] # BGRA
    mask_image = alpha / 255 # 0~255 -> 255로 나누면 0~1 사이의 값 ( 1: 불투명, 0: 완전 )
    
    for c in range(0, 3): #channel BGR
        image[y-h:y+h, x-w:x+w, c] = (overlay_image[:, :, c] * mask_image) + (image[y-h:y+h, x-w:x+w, c] * (1 - mask_image))
        


# 얼굴을 찾고, 찾은 얼굴에 표시를 해주기 위한 변수 정의
mp_face_detection = mp.solutions.face_detection # 얼굴 검출을 위한 face_detection 모듈을 사용
mp_drawing = mp.solutions.drawing_utils  # 얼굴의 특징을 그리기 위한 drawing_utils 모듈을 사용

# 이미지 파일의 경우 이것을 사용하세요:
# IMAGE_FILES = []
# with mp_face_detection.FaceDetection(
#     model_selection=1, min_detection_confidence=0.5) as face_detection:
#   for idx, file in enumerate(IMAGE_FILES):
#     image = cv2.imread(file)
#     # 작업 전에 BGR 이미지를 RGB로 변환합니다.
#     results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
# 	# 이미지를 출력하고 그 위에 얼굴 박스를 그립니다.
#     if not results.detections:
#       continue
#     annotated_image = image.copy()
#     for detection in results.detections:
#       print('Nose tip:')
#       print(mp_face_detection.get_key_point(
#           detection, mp_face_detection.FaceKeyPoint.NOSE_TIP))
#       mp_drawing.draw_detection(annotated_image, detection)
#     cv2.imwrite('/tmp/annotated_image' + str(idx) + '.png', annotated_image)

# 웹캠, 영상 파일의 경우 이것을 사용하세요.:
# cap = cv2.VideoCapture(0)

# 이미지 불러오기 (캐릭터 이미지)
image_right_eye = cv2.imread('right_eye.png', cv2.IMREAD_UNCHANGED) # 100x100  # cv2.IMREAD_UNCHANGED --> 투명이미지 (벡터이미지)
image_left_eye = cv2.imread('left_eye.png', cv2.IMREAD_UNCHANGED)  # 100x100
image_nose = cv2.imread('nose.png', cv2.IMREAD_UNCHANGED)  # 300x100

cap = cv2.VideoCapture('face_video.mp4') # 동영상 파일열기
run_flg = False

# model_selection 값 - 0 : 카메라로 부터 2m 이내의 근거리 / 1 : 카메라로 부터  5m 이내의 얼굴에 적합 
# min_detection_confidence - 0~1사이의 수 : 얼굴인지 검출하는 정확도 (신뢰도)

with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            if run_flg:
                print("Video end ")
            else:
                print("Not found video")
            # 비디오 파일의 경우 'continue'를 사용하시고, 웹캠에 경우에는 'break'를 사용하세요.
            break

        # 보기 편하기 위해 이미지를 좌우를 반전하고, BGR 이미지를 RGB로 변환합니다.
        # image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # BGR -> RGB
        # print(">> image : {} ".format(image))
        # 성능을 향상시키려면 이미지를 작성 여부를 False으로 설정하세요.
        image.flags.writeable = False
        results = face_detection.process(image)

        # 영상에 얼굴 감지 주석 그리기 기본값 : True.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.detections: # 검출된 얼굴이 있다면.. 
            # 6개 특징 : 오른쪽 눈, 왼쪽 눈, 코 끝부분, 입 중심, 오른쪽 귀, 왼쪽 귀 (귀구슬점, 이주)
            for detection in results.detections:
                # mp_drawing.draw_detection(image, detection) # 사각형을 그려준다.
                # print(">> detection : {} ".format(detection))
                
                # 특정 위치 가져오기 
                keypoints = detection.location_data.relative_keypoints
                right_eye = keypoints[0] # 오른쪽 눈
                left_eye = keypoints[1] # 왼쪽 눈
                nose_tip = keypoints[2] # 코 끝부분
                # print(">> right_eye : {}, left_eye : {}, nose_tip : {}".format(right_eye, left_eye, nose_tip))
                
                h, w, _ = image.shape # height, width, channel : 이미지로 부터 세로, 가로 크기 가져옴
                # 이미지 내에서 실제 좌표 (x, y)
                right_eye = (int(right_eye.x * w) - 20, int(right_eye.y * h) - 100) # 해당 좌표에서 -20(width) , -100(height) 약간옆에 표시
                left_eye = (int(left_eye.x * w) + 20, int(left_eye.y * h) - 100) 
                nose_tip = (int(nose_tip.x * w), int(nose_tip.y * h)) 
                
                # # 양 눈에 동그라미 그리기
                # cv2.circle(image, right_eye, 50, (255, 0, 0), 10, cv2.LINE_AA) # 파란색
                # cv2.circle(image, left_eye, 50, (0, 255, 0), 10, cv2.LINE_AA) # 초록색
                # # 코에 동그라미 그리기
                # cv2.circle(image, nose_tip, 50, (0, 255, 255), 10, cv2.LINE_AA) # 노란색

                # # 각 특징에다가 가져온 이미지 씌우기
                # image[right_eye[1]-50:right_eye[1]+50, right_eye[0]-50:right_eye[0]+50] = image_right_eye
                # image[left_eye[1]-50:left_eye[1]+50, left_eye[0]-50:left_eye[0]+50] = image_left_eye
                # image[nose_tip[1]-50:nose_tip[1]+50, nose_tip[0]-150:nose_tip[0]+150] = image_nose

                # 각 특징에다가 가져온 이미지 (투명하게 변경해서) 씌우기
                overlay(image, *right_eye, 50, 50, image_right_eye)
                overlay(image, *left_eye, 50, 50, image_left_eye)
                overlay(image, *nose_tip, 150, 50, image_nose)

        # cv2.imshow('MediaPipe Face Detection', image)
        cv2.imshow('MediaPipe Face Detection', cv2.resize(image, None, fx=0.5, fy=0.5))
        run_flg = True
        # if cv2.waitKey(5) & 0xFF == 27:
        if cv2.waitKey(1) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()



### python.readthedocs 
### https://opencv-python.readthedocs.io/en/latest/doc/29.matchDigits/matchDigits.html 
