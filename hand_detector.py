import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

# 下載模型檔案（只需執行一次）
import urllib.request
import os

model_path = "hand_landmarker.task"
if not os.path.exists(model_path):
    print("下載 MediaPipe 手部模型...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        model_path
    )
    print("下載完成！")

# 初始化 Hand Landmarker
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7
)
detector = vision.HandLandmarker.create_from_options(options)

# 繪製連線定義
CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17)
]

cap = cv2.VideoCapture(0)
print("攝影機已開啟，按 Q 退出")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 轉成 RGB 給 MediaPipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)

    h, w, _ = frame.shape

    if result.hand_landmarks:
        for hand in result.hand_landmarks:
            landmarks = []
            # 畫關鍵點
            for lm in hand:
                landmarks.extend([lm.x, lm.y, lm.z])
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
            landmark_array = np.array(landmarks)  # shape: (63,)
            cv2.putText(frame, f"Landmarks: {landmark_array.shape}", (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            # 畫連線
            for conn in CONNECTIONS:
                a = hand[conn[0]]
                b = hand[conn[1]]
                ax, ay = int(a.x * w), int(a.y * h)
                bx, by = int(b.x * w), int(b.y * h)
                cv2.line(frame, (ax, ay), (bx, by), (0, 200, 255), 2)

        cv2.putText(frame, "Hand Detected!", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Hand Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()