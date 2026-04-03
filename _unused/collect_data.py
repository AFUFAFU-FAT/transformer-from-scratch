import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import csv
import os
import time

# 初始化 MediaPipe
model_path = "hand_landmarker.task"
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7
)
detector = vision.HandLandmarker.create_from_options(options)

CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17)
]

# 設定要收集的手語類別
# 先從幾個簡單的 ASL 字母開始測試
LABELS = ["A", "B", "C", "D", "E", "nothing"]

def collect_data():
    os.makedirs("data", exist_ok=True)
    csv_path = "data/landmarks.csv"

    # 建立 CSV 標頭
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            header = [f"x{i}" for i in range(21)] + \
                     [f"y{i}" for i in range(21)] + \
                     [f"z{i}" for i in range(21)] + ["label"]
            writer.writerow(header)

    cap = cv2.VideoCapture(0)
    current_label = LABELS[0]
    collecting = False
    count = 0
    target = 100  # 每個類別收集 100 筆

    print(f"按數字鍵切換類別：")
    for i, label in enumerate(LABELS):
        print(f"  {i} → {label}")
    print("按 SPACE 開始/停止收集，按 Q 退出")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_image)

        h, w, _ = frame.shape

        if result.hand_landmarks:
            for hand in result.hand_landmarks:
                landmarks = []
                for lm in hand:
                    landmarks.extend([lm.x, lm.y, lm.z])
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

                for conn in CONNECTIONS:
                    a = hand[conn[0]]
                    b = hand[conn[1]]
                    cv2.line(frame,
                             (int(a.x*w), int(a.y*h)),
                             (int(b.x*w), int(b.y*h)),
                             (0, 200, 255), 2)

                # 收集資料
                if collecting and len(landmarks) == 63:
                    with open(csv_path, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(landmarks + [current_label])
                    count += 1
                    if count >= target:
                        collecting = False
                        count = 0
                        print(f"✅ {current_label} 收集完成！")

        # UI 顯示
        color = (0, 0, 255) if collecting else (255, 255, 255)
        cv2.putText(frame, f"Label: {current_label}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, f"Count: {count}/{target}", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        status = "RECORDING" if collecting else "READY (SPACE to start)"
        cv2.putText(frame, status, (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("Data Collection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            collecting = not collecting
            count = 0
        elif key in [ord(str(i)) for i in range(len(LABELS))]:
            current_label = LABELS[int(chr(key))]
            collecting = False
            count = 0
            print(f"切換到類別：{current_label}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n資料已儲存到 {csv_path}")

if __name__ == "__main__":
    collect_data()