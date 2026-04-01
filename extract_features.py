import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import json
import csv
import os

model_path = "hand_landmarker.task"
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,  # 改用影片模式
    num_hands=1,
    min_hand_detection_confidence=0.3,      # 降低門檻
    min_hand_presence_confidence=0.3,
    min_tracking_confidence=0.3
)

def encode_location(location):
    locations = [
        "401_in_front_of_body", "402_chest", "403_face",
        "404_head", "405_side_of_body", "406_waist", None
    ]
    if location in locations:
        return locations.index(location)
    return len(locations) - 1

def encode_handshape(handshape):
    shapes = [
        "twoFinger", "oneFingerPoint", "fist", "openHand",
        "thumbUp", "pinch", "claw", None
    ]
    if handshape in shapes:
        return shapes.index(handshape)
    return len(shapes) - 1

def process_video(video_path, catalog_info):
    """用影片串流模式提取關鍵點"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        return []

    detector = vision.HandLandmarker.create_from_options(options)

    loc_code = encode_location(catalog_info.get("location1"))
    hs_code  = encode_handshape(catalog_info.get("handshape1"))
    stroke   = catalog_info.get("stroke") or 0

    samples = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        timestamp_ms = int(frame_idx * 1000 / fps)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect_for_video(mp_image, timestamp_ms)

        if result.hand_landmarks:
            hand = result.hand_landmarks[0]
            landmarks = []
            for lm in hand:
                landmarks.extend([lm.x, lm.y, lm.z])
            feature = np.append(landmarks, [loc_code, hs_code, stroke])
            samples.append(feature)

        frame_idx += 1

    cap.release()
    detector.close()
    return samples

# ============================================================
with open("data/catalog_full.json", encoding="utf-8") as f:
    catalog = json.load(f)

csv_path = "data/features.csv"
header = [f"lm_{i}" for i in range(63)] + ["location", "handshape", "stroke", "label"]

total_samples = 0
zero_count = 0

with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(header)

    for i, item in enumerate(catalog):
        video_file = f"videos/{item['video_file']}"
        if not os.path.exists(video_file):
            continue

        samples = process_video(video_file, item)
        label = item["name"]

        for s in samples:
            writer.writerow(list(s) + [label])

        total_samples += len(samples)
        if len(samples) == 0:
            zero_count += 1

        if (i + 1) % 100 == 0:
            print(f"進度：{i+1}/{len(catalog)}，目前共 {total_samples} 筆")

print(f"\n完成！共提取 {total_samples} 筆特徵")
print(f"0 筆的詞彙：{zero_count} 個")
print(f"已儲存到 {csv_path}")