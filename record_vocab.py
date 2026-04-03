"""
record_vocab.py v2
對著攝影機自錄手語詞彙，建立零 Domain Gap 的訓練資料

特徵維度：81 維
  - 手部關鍵點：21 個 × (x,y,z) = 63 維
  - 相對身體位置：手腕相對肩膀中心的正規化座標 (rx, ry, rz) = 3 維
  - 手部與身體錨點：鼻子、左肩、右肩、左髖、右髖 × (x,y) = 10 維  (只取x,y不取z)
  - 結構化特徵：3 維（預留，填 0）
  - 手掌朝向補充：2 維（手腕到中指根部的向量方向）= 2 維
  → 63 + 3 + 10 + 3 + 2 = 81 維

按鍵：空白=錄製  N=下一個  R=重錄  Q=退出

使用方式：
  python record_vocab.py            # 全部 63 個詞彙
  python record_vocab.py --unit U1  # 只錄 U1
  python record_vocab.py --reps 5   # 每詞彙 5 次（預設 8）
  python record_vocab.py --start 你  # 從指定詞彙開始
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import pandas as pd
import os
import time
import argparse
import urllib.request
from collections import deque
from PIL import Image, ImageDraw, ImageFont

# ────────────────────────────────────────────────────────────
#  詞彙清單（部編教材 U1~U3，共 63 個）
# ────────────────────────────────────────────────────────────
VOCAB = {
    "U1": [
        # 人稱
        "你", "我", "他", "我們",
        # 社交
        "好", "謝謝", "對不起", "再見",
        # 核心動詞
        "喜歡", "不喜歡", "想", "是", "不是", "學", "找",
        # 情態
        "開心", "認真",
        # 關係
        "一樣", "嗎",
        # TSL 身份
        # "聾人",  # 暫停用：與「是」手勢特徵太相似，持續混淆
        "聽人",
        # 時間
        "現在", "晚上",
        # ── 暫時停用（課堂指令/情境窄）──
        # "起床", "安靜", "上課", "放學", "坐下", "缺席",
        # ── 暫時停用（時間詞冗餘）──
        # "早上", "中午", "下午",
        # ── 暫時停用（抽象/低頻）──
        # "希望", "相見", "找不到", "適合",
        # ── 暫時停用（meta 詞）──
        # "手語",
        # ── 暫時停用（可由其他詞組合）──
        # "不一樣",  # = 不是 + 一樣
        # "今天" 手語與「現在」相同，合併為同一類別（見 LABEL_ALIASES）
        # "平安" 手語與「安靜」相同，合併為同一類別（見 LABEL_ALIASES）
    ],
    "U2": [
        # 動詞
        "介紹", "問", "說", "住",
        # 疑問
        "甚麼",
        # 地點
        "地方", "家",
        # ── 暫時停用（可用「我」替代）──
        # "自己",
        # ── 暫時停用（meta 詞）──
        # "手語名字",
        # ── 暫時停用（情境窄）──
        # "以前", "出生", "活動",
        # "見面" 手語與「相見」相同，合併為同一類別（見 LABEL_ALIASES）
        # "彼此" 為複合手語：我們 + 一樣（見 COMPOUND_SIGNS）
        # "哪裡" 為複合手語：甚麼 + 地方（見 COMPOUND_SIGNS）
        # "家庭活動" 為複合手語：家 + 活動（見 COMPOUND_SIGNS）
    ],
    "U3": [
        # 人物
        "人", "爸爸", "媽媽",
        # ── 暫時停用（家庭成員太多）──
        # "爺爺", "奶奶", "哥哥", "弟弟", "姊姊", "妹妹",
        # ── 暫時停用（性別詞）──
        # "男", "女",
        # ── 暫時停用（量詞/比較詞）──
        # "多少", "全部", "像",
        # "家" 已移至 U2 錄製
        # "家人" 為複合手語：家 + 人（見 COMPOUND_SIGNS）
    ],
}

ALL_VOCAB = VOCAB["U1"] + VOCAB["U2"] + VOCAB["U3"]  # 30 個（精簡版，暫停用詞已保留在 VOCAB 內以 # 標記）

# 預測時的顯示別名：key 會被模型輸出，value 是補充說明
LABEL_ALIASES = {
    "現在": "現在／今天",
    "安靜": "安靜／平安",
    "相見": "相見／見面",
}

# 複合手語：由多個已知詞彙組合而成，不需單獨錄製
COMPOUND_SIGNS = {
    "彼此":   ["我們", "一樣"],
    "一言為定": ["說",  "一樣"],
    "哪裡":   ["甚麼", "地方"],
    "家庭活動": ["家",  "活動"],
    "家人":   ["家",  "人"],
}

# ────────────────────────────────────────────────────────────
#  設定
# ────────────────────────────────────────────────────────────
HAND_MODEL = "hand_landmarker.task"
POSE_MODEL = "pose_landmarker_lite.task"
OUTPUT_CSV = "data/recorded_features.csv"

RECORD_SECONDS    = 3.0
COUNTDOWN_SECONDS = 3
FEAT_DIM          = 167   # 右手 68 + 左手 68 + 身體錨點 10 + 結構 3 + 雙臂 12 + 手臉距離 6

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17)
]

# Pose landmark 索引
NOSE         = 0
MOUTH_L      = 9   # 嘴角左
MOUTH_R      = 10  # 嘴角右
L_EAR        = 7
R_EAR        = 8
L_SHOULDER   = 11
R_SHOULDER   = 12
L_ELBOW      = 13
R_ELBOW      = 14
L_WRIST_POSE = 15
R_WRIST_POSE = 16
L_HIP        = 23
R_HIP        = 24

# ────────────────────────────────────────────────────────────
#  下載模型檔
# ────────────────────────────────────────────────────────────
def download_model(path, url):
    if not os.path.exists(path):
        print(f"下載 {path} ...")
        urllib.request.urlretrieve(url, path)
        print(f"✅ 下載完成：{path}")

download_model(
    HAND_MODEL,
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
download_model(
    POSE_MODEL,
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)

# ────────────────────────────────────────────────────────────
#  初始化 MediaPipe（與 hand_detector.py 同結構）
# ────────────────────────────────────────────────────────────
hand_detector = vision.HandLandmarker.create_from_options(
    vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=HAND_MODEL),
        num_hands=2,
        min_hand_detection_confidence=0.4,
        min_hand_presence_confidence=0.4,
        min_tracking_confidence=0.4,
    )
)

pose_detector = vision.PoseLandmarker.create_from_options(
    vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=POSE_MODEL),
        min_pose_detection_confidence=0.4,
        min_pose_presence_confidence=0.4,
        min_tracking_confidence=0.4,
    )
)
print("✅ MediaPipe 初始化完成（Hand + Pose）")

# ────────────────────────────────────────────────────────────
#  中文字型
# ────────────────────────────────────────────────────────────
FONT_PATH = None
for p in ["C:/Windows/Fonts/msjh.ttc", "C:/Windows/Fonts/mingliu.ttc", "C:/Windows/Fonts/kaiu.ttf"]:
    if os.path.exists(p):
        FONT_PATH = p
        break

def put_chinese(img_bgr, text, pos, size=26, color=(255, 255, 255)):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw    = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.truetype(FONT_PATH, size) if FONT_PATH else ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()
    draw.text(pos, text, font=font, fill=color)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

# ────────────────────────────────────────────────────────────
#  特徵提取（81 維）
# ────────────────────────────────────────────────────────────
def extract_features(frame_bgr):
    """
    同時跑 Hand（最多 2 手）+ Pose，回傳 149 維特徵向量。
      右手 68 維（lm 0-67）+ 左手 68 維（68-135）
      + 身體錨點 10 維 + 結構 3 維
    畫面已 flip，MediaPipe "Left" 對應螢幕右手，"Right" 對應螢幕左手。
    若完全未偵測到手 → 回傳 None。
    """
    rgb    = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    hand_result = hand_detector.detect(mp_img)
    pose_result = pose_detector.detect(mp_img)

    if not hand_result.hand_landmarks:
        return None, hand_result, pose_result

    # ── 分左右手（flip 後 MediaPipe Left = 畫面右手）──
    right_hand = None
    left_hand  = None
    for i, cls in enumerate(hand_result.handedness):
        label = cls[0].category_name  # "Left" or "Right"
        if label == "Left":
            right_hand = hand_result.hand_landmarks[i]
        else:
            left_hand  = hand_result.hand_landmarks[i]

    # ── Pose 錨點 ──
    if pose_result.pose_landmarks:
        pose  = pose_result.pose_landmarks[0]
        nose  = np.array([pose[NOSE].x,      pose[NOSE].y],      dtype=np.float32)
        l_sho = np.array([pose[L_SHOULDER].x, pose[L_SHOULDER].y], dtype=np.float32)
        r_sho = np.array([pose[R_SHOULDER].x, pose[R_SHOULDER].y], dtype=np.float32)
        l_hip = np.array([pose[L_HIP].x,      pose[L_HIP].y],      dtype=np.float32)
        r_hip = np.array([pose[R_HIP].x,      pose[R_HIP].y],      dtype=np.float32)
        mid_sho   = (l_sho + r_sho) / 2.0
        sho_width = max(abs(r_sho[0] - l_sho[0]), 1e-6)
        body_anchors = np.concatenate([nose, l_sho, r_sho, l_hip, r_hip]).astype(np.float32)
        r_elbow_lm   = pose[R_ELBOW];      l_elbow_lm   = pose[L_ELBOW]
        r_wrist_p_lm = pose[R_WRIST_POSE]; l_wrist_p_lm = pose[L_WRIST_POSE]
        r_sho_lm     = pose[R_SHOULDER];   l_sho_lm     = pose[L_SHOULDER]
        mouth_c  = np.array([(pose[MOUTH_L].x + pose[MOUTH_R].x) / 2,
                              (pose[MOUTH_L].y + pose[MOUTH_R].y) / 2], dtype=np.float32)
        r_ear_pt = np.array([pose[R_EAR].x, pose[R_EAR].y], dtype=np.float32)
        l_ear_pt = np.array([pose[L_EAR].x, pose[L_EAR].y], dtype=np.float32)
    else:
        nose         = np.array([0.5, 0.3], dtype=np.float32)
        mid_sho      = np.array([0.5, 0.5], dtype=np.float32)
        sho_width    = 0.3
        body_anchors = np.zeros(10, dtype=np.float32)
        r_elbow_lm = l_elbow_lm = r_wrist_p_lm = l_wrist_p_lm = None
        r_sho_lm   = l_sho_lm  = None
        mouth_c  = np.zeros(2, dtype=np.float32)
        r_ear_pt = np.zeros(2, dtype=np.float32)
        l_ear_pt = np.zeros(2, dtype=np.float32)

    def hand_feat(hand):
        """68 維：63 landmark + 3 relative wrist + 2 palm dir"""
        if hand is None:
            return np.zeros(68, dtype=np.float32)
        coords = np.array([[lm.x, lm.y, lm.z] for lm in hand], dtype=np.float32).flatten()
        wrist  = np.array([hand[0].x, hand[0].y, hand[0].z], dtype=np.float32)
        rel_w  = np.array([
            (wrist[0] - mid_sho[0]) / sho_width,
            (wrist[1] - mid_sho[1]) / sho_width,
            wrist[2],
        ], dtype=np.float32)
        pd = np.array([hand[9].x - hand[0].x, hand[9].y - hand[0].y], dtype=np.float32)
        pd = pd / (np.linalg.norm(pd) + 1e-6)
        return np.concatenate([coords, rel_w, pd])  # 63+3+2 = 68

    def arm_feat(elbow_lm, wrist_lm, sho_lm):
        """6 維：手肘相對位置(2) + 上臂方向(2) + 前臂方向(2)"""
        if elbow_lm is None or sho_lm is None:
            return np.zeros(6, dtype=np.float32)
        elbow = np.array([elbow_lm.x, elbow_lm.y], dtype=np.float32)
        sho   = np.array([sho_lm.x,   sho_lm.y],   dtype=np.float32)
        elbow_rel = np.array([(elbow[0] - mid_sho[0]) / sho_width,
                               (elbow[1] - mid_sho[1]) / sho_width], dtype=np.float32)
        upper = elbow - sho
        upper = upper / (np.linalg.norm(upper) + 1e-6)
        if wrist_lm is not None:
            wrist   = np.array([wrist_lm.x, wrist_lm.y], dtype=np.float32)
            forearm = wrist - elbow
            forearm = forearm / (np.linalg.norm(forearm) + 1e-6)
        else:
            forearm = np.zeros(2, dtype=np.float32)
        return np.concatenate([elbow_rel, upper, forearm])

    def face_dist_feat(hand, ear_pt):
        """3 維：手腕到鼻子/嘴巴/耳朵的正規化距離（以肩寬為單位）"""
        if hand is None:
            return np.zeros(3, dtype=np.float32)
        wrist = np.array([hand[0].x, hand[0].y], dtype=np.float32)
        d_nose  = np.linalg.norm(wrist - nose)    / sho_width
        d_mouth = np.linalg.norm(wrist - mouth_c) / sho_width
        d_ear   = np.linalg.norm(wrist - ear_pt)  / sho_width
        return np.array([d_nose, d_mouth, d_ear], dtype=np.float32)

    struct = np.zeros(3, dtype=np.float32)
    feat   = np.concatenate([
        hand_feat(right_hand), hand_feat(left_hand), body_anchors, struct,
        arm_feat(r_elbow_lm, r_wrist_p_lm, r_sho_lm),
        arm_feat(l_elbow_lm, l_wrist_p_lm, l_sho_lm),
        face_dist_feat(right_hand, r_ear_pt),
        face_dist_feat(left_hand,  l_ear_pt),
    ])
    assert len(feat) == FEAT_DIM, f"特徵維度錯誤：{len(feat)} != {FEAT_DIM}"

    return feat, hand_result, pose_result


def draw_hand_pose(frame, hand_result, pose_result):
    h, w = frame.shape[:2]

    # 畫手部（支援雙手）
    if hand_result.hand_landmarks:
        for hand in hand_result.hand_landmarks:
            for lm in hand:
                cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 5, (0, 255, 0), -1)
            for a, b in HAND_CONNECTIONS:
                pa, pb = hand[a], hand[b]
                cv2.line(frame,
                         (int(pa.x * w), int(pa.y * h)),
                         (int(pb.x * w), int(pb.y * h)),
                         (0, 200, 255), 2)

    # 畫 Pose 錨點
    if pose_result.pose_landmarks:
        pose = pose_result.pose_landmarks[0]
        for idx in [NOSE, L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_HIP, R_HIP]:
            lm = pose[idx]
            cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 7, (255, 100, 0), -1)
        # 連肩膀 + 手臂
        for a_idx, b_idx in [(L_SHOULDER, R_SHOULDER),
                              (R_SHOULDER, R_ELBOW), (L_SHOULDER, L_ELBOW)]:
            a, b = pose[a_idx], pose[b_idx]
            cv2.line(frame,
                     (int(a.x * w), int(a.y * h)),
                     (int(b.x * w), int(b.y * h)),
                     (255, 100, 0), 2)

    return frame


# ────────────────────────────────────────────────────────────
#  儲存
# ────────────────────────────────────────────────────────────
def save_features(features_list, labels_list, path):
    cols   = [f"lm_{i}" for i in range(FEAT_DIM)] + ["label"]
    rows   = [list(f) + [l] for f, l in zip(features_list, labels_list)]
    df_new = pd.DataFrame(rows, columns=cols)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if os.path.exists(path):
        df_old = pd.read_csv(path)
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new

    df.to_csv(path, index=False)
    return len(df)


# ────────────────────────────────────────────────────────────
#  主程式
# ────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit",  type=str, default="ALL",
                        help="要錄製的單元：U1 / U2 / U3 / ALL（預設 ALL）")
    parser.add_argument("--reps",  type=int, default=8,
                        help="每個詞彙錄幾次（預設 8）")
    parser.add_argument("--start", type=str, default="",
                        help="從指定詞彙開始（例：--start 今天）")
    parser.add_argument("--label", type=str, default="",
                        help="錄製任意自訂標籤（例：--label _unknown_ --reps 20）")
    parser.add_argument("--output", type=str, default="",
                        help="輸出 CSV 路徑（預設 data/recorded_features.csv）；錄測試集時用 data/test_features.csv")
    args = parser.parse_args()

    output_csv = args.output if args.output else OUTPUT_CSV

    # --label 模式：錄製任意標籤（負樣本、補錄等）
    if args.label:
        vocab     = [args.label]
        start_idx = 0
        reps      = args.reps
        print(f"\n🎥 自訂標籤錄製模式")
        print(f"   標籤：{args.label}  次數：{reps}")
        if args.label == "_unknown_":
            print(f"\n   請錄製以下任意動作（每次 3 秒）：")
            print(f"   • 手放在身體兩側或桌面（靜止）")
            print(f"   • 隨機揮手、指向、開合拳頭")
            print(f"   • 詞彙之間的過渡動作")
            print(f"   • 抓頭、摸臉、伸懶腰等日常動作")
            print(f"   目標：讓模型學會說「我不知道」")
        print(f"   輸出：{output_csv}")
    else:
        # 決定詞彙清單
        if args.unit.upper() == "ALL":
            vocab = ALL_VOCAB
        elif args.unit.upper() in VOCAB:
            vocab = VOCAB[args.unit.upper()]
        else:
            print(f"❌ 未知單元：{args.unit}，可選 U1 / U2 / U3 / ALL")
            return

        # 從指定詞彙開始
        start_idx = 0
        if args.start and args.start in vocab:
            start_idx = vocab.index(args.start)
            print(f"從「{args.start}」開始（第 {start_idx+1} 個）")

        reps = args.reps
        print(f"\n🎥 手語詞彙錄製工具 v2（81 維特徵）")
        print(f"   詞彙數：{len(vocab)}  每詞彙：{reps} 次  預計時間：{len(vocab[start_idx:]) * reps * 11 // 60} 分鐘")
        print(f"   輸出：{output_csv}")
        print(f"\n   查詢手語打法：https://special.moe.gov.tw/signlanguage")
    print(f"\n   空白=開始錄製  N=下一個  R=重錄當前詞彙  Q=儲存並退出")
    print("-" * 55)

    # 開啟攝影機
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print("❌ 無法開啟攝影機")
        return
    for _ in range(5):
        ret, _ = cap.read()
        if ret:
            break

    all_features = []
    all_labels   = []
    vocab_idx    = start_idx
    rep_count    = 0
    state        = "READY"   # READY / COUNTDOWN / RECORDING / DONE
    state_start  = 0.0
    rec_frames   = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        now   = time.time()

        # 全部完成
        if vocab_idx >= len(vocab):
            frame = put_chinese(frame, "全部詞彙錄製完成！按 Q 儲存退出",
                                (60, 220), size=28, color=(0, 255, 100))
            cv2.imshow("手語錄製", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        current_word = vocab[vocab_idx]

        # 特徵提取
        feat, hand_res, pose_res = extract_features(frame)
        frame = draw_hand_pose(frame, hand_res, pose_res)
        hand_ok = feat is not None

        # ── 狀態機 ──
        if state == "COUNTDOWN":
            elapsed = now - state_start
            remain  = COUNTDOWN_SECONDS - elapsed
            num     = int(remain) + 1
            frame = put_chinese(frame, str(num), (295, 160), size=100,
                                color=(0, 255, 255) if num > 1 else (0, 80, 255))
            if elapsed >= COUNTDOWN_SECONDS:
                state       = "RECORDING"
                state_start = now
                rec_frames  = []

        elif state == "RECORDING":
            elapsed = now - state_start
            remain  = max(0, RECORD_SECONDS - elapsed)

            if hand_ok:
                rec_frames.append(feat.copy())

            # 錄製進度條
            prog = int((elapsed / RECORD_SECONDS) * 620)
            cv2.rectangle(frame, (10, 460), (10 + prog, 475), (0, 80, 255), -1)
            cv2.rectangle(frame, (10, 460), (630, 475), (100, 100, 100), 1)
            cv2.circle(frame, (620, 25), 12, (0, 0, 255), -1)  # 錄製燈
            frame = put_chinese(frame, f"錄製中  {remain:.1f}s  已錄 {len(rec_frames)} 幀",
                                (30, 390), size=24, color=(0, 80, 255))

            if elapsed >= RECORD_SECONDS:
                if len(rec_frames) >= 10:
                    all_features.extend(rec_frames)
                    all_labels.extend([current_word] * len(rec_frames))
                    rep_count += 1
                    print(f"  ✅ [{current_word}] {rep_count}/{reps}次 — {len(rec_frames)} 幀")
                else:
                    print(f"  ⚠️  [{current_word}] 手部幀數不足（{len(rec_frames)} 幀），請重錄")
                rec_frames = []
                state = "DONE" if rep_count >= reps else "READY"

        elif state == "DONE":
            frame = put_chinese(frame, f"✅ 完成！按 N 繼續",
                                (160, 390), size=26, color=(0, 255, 100))

        # ── UI ──
        # 進度列（頂部）
        prog_ratio = vocab_idx / len(vocab)
        cv2.rectangle(frame, (0, 0), (int(640 * prog_ratio), 6), (0, 200, 100), -1)
        cv2.rectangle(frame, (0, 0), (640, 6), (80, 80, 80), 1)

        # 詞彙 + 進度
        frame = put_chinese(frame, current_word, (20, 12), size=44, color=(255, 255, 100))
        frame = put_chinese(frame,
                            f"{vocab_idx+1}/{len(vocab)}  第 {rep_count}/{reps} 次",
                            (20, 62), size=20, color=(180, 180, 180))

        # 手部狀態
        n_hands = len(hand_res.hand_landmarks) if hand_res and hand_res.hand_landmarks else 0
        hand_color = (0, 255, 0) if hand_ok else (0, 60, 255)
        hand_text  = f"手部偵測 ✓ ({n_hands} 手)" if hand_ok else "請將手放入畫面"
        frame = put_chinese(frame, hand_text, (20, 90), size=20, color=hand_color)

        # Pose 狀態
        pose_ok    = bool(pose_res.pose_landmarks) if pose_res else False
        pose_color = (0, 200, 100) if pose_ok else (100, 100, 100)
        frame = put_chinese(frame, "身體偵測 ✓" if pose_ok else "身體未偵測（相對位置將填 0）",
                            (20, 115), size=18, color=pose_color)

        if state == "READY":
            frame = put_chinese(frame, "按 空白 開始錄製",
                                (170, 390), size=26, color=(200, 200, 200))

        frame = put_chinese(frame, "空白:錄製  N:下一個  R:重錄  Q:退出",
                            (20, 430), size=17, color=(130, 130, 130))

        cv2.imshow("手語錄製 v2", frame)

        # ── 鍵盤 ──
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord(' ') and state in ("READY",):
            state       = "COUNTDOWN"
            state_start = now
            print(f"\n  [{current_word}] 第 {rep_count+1}/{reps} 次...")

        elif key == ord('n'):
            if rep_count > 0 or state == "DONE":
                vocab_idx += 1
                rep_count  = 0
                state      = "READY"
                rec_frames = []

        elif key == ord('r'):
            # 清除當前詞彙已錄資料
            before = len(all_labels)
            all_features = [f for f, l in zip(all_features, all_labels) if l != current_word]
            all_labels   = [l for l in all_labels if l != current_word]
            print(f"  🗑️  清除 {current_word} 的 {before - len(all_labels)} 幀，重新錄製")
            rep_count  = 0
            state      = "READY"
            rec_frames = []

    cap.release()
    cv2.destroyAllWindows()

    # ── 儲存 ──
    if all_features:
        total = save_features(all_features, all_labels, output_csv)
        unique = len(set(all_labels))
        print(f"\n✅ 儲存完成")
        print(f"   新增：{len(all_features)} 幀  {unique} 個詞彙")
        print(f"   累計：{total} 筆  →  {output_csv}")
        if output_csv != OUTPUT_CSV:
            print(f"\n測試集錄製完成，接著重新建立序列並重新訓練：")
        else:
            print(f"\n接著執行：")
        print(f"  python build_sequences.py")
        print(f"  python train_lstm.py")
    else:
        print("\n沒有錄製任何資料")


if __name__ == "__main__":
    main()