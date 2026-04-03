"""
recognize.py
台灣手語即時辨識系統

結構與 hand_detector.py 完全一致（避免 Windows 執行緒死鎖）：
  - MediaPipe HandLandmarker IMAGE 模式
  - 主迴圈：cap.read() → detector.detect() → BiLSTM 推論
  - Qwen 語意修正（背景執行緒，不影響攝影機）

按鍵：Q=退出  C=清空詞彙  S=手動送出句子  空白=暫停
"""

import cv2
import math
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import torch
import torch.nn as nn
import pickle
import threading
import time
import os
from collections import deque
from PIL import Image, ImageDraw, ImageFont

# ────────────────────────────────────────────────────────────
#  模型定義（與 train_lstm.py 完全一致）
# ────────────────────────────────────────────────────────────
class SignLanguageBiLSTM(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=64, num_layers=1, dropout=0.6):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(
            input_size=64, hidden_size=hidden_dim,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )
        lstm_out_dim = hidden_dim * 2
        self.attention  = nn.Linear(lstm_out_dim, 1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_out_dim, num_classes),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.input_proj(x)
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out)
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = (attn_weights * lstm_out).sum(dim=1)
        return self.classifier(context)

# ────────────────────────────────────────────────────────────
#  設定
# ────────────────────────────────────────────────────────────
MODEL_PATH      = "models/lstm_best.pth"
CONFIG_PATH     = "models/lstm_config.pkl"
LABEL_PATH      = "data/seq_label_encoder.pkl"
HAND_MODEL      = "hand_landmarker.task"
POSE_MODEL      = "pose_landmarker_lite.task"

FEAT_DIM    = 179   # 167 原始 + 2 身體朝向 + 10 手指伸展分數

W_HANDSHAPE = 6.0
W_FINGERTIP = 4.0
W_FINGER    = 3.0
W_PALM_DIR  = 3.0
W_POSITION  = 2.5
W_ARM       = 2.5
W_FACE      = 3.0
W_ORIENT    = 2.0
W_ANCHOR    = 1.5
FINGERTIPS  = {4, 8, 12, 16, 20}

TIP_IDX = [4,  8, 12, 16, 20]
MCP_IDX = [1,  5,  9, 13, 17]

def compute_finger_extensions(lm63):
    """
    單幀：63 維手部座標 → 5 指伸展分數。
    lm63: shape (63,)
    伸直 ≈ 2~3，彎曲 ≈ 1，握拳所有指 ≈ 1
    """
    lms   = lm63.reshape(21, 3)
    wrist = lms[0]
    scores = np.empty(5, dtype=np.float32)
    for fi, (ti, mi) in enumerate(zip(TIP_IDX, MCP_IDX)):
        mcp_d = np.linalg.norm(lms[mi] - wrist) + 1e-6
        scores[fi] = np.linalg.norm(lms[ti] - wrist) / mcp_d
    return scores

def apply_feature_weights(feat):
    """與 build_sequences.py 完全一致（單幀，shape: (159,)）"""
    out = feat.copy()
    for offset in (0, 68):
        wx, wy = feat[offset + 0], feat[offset + 1]
        mx, my = feat[offset + 27], feat[offset + 28]
        scale = np.sqrt((mx - wx) ** 2 + (my - wy) ** 2) + 1e-6
        for i in range(21):
            b = offset + i * 3
            w = W_FINGERTIP if i in FINGERTIPS else W_FINGER
            out[b + 0] = (feat[b + 0] - wx) / scale * w
            out[b + 1] = (feat[b + 1] - wy) / scale * w
            out[b + 2] = feat[b + 2] * w
        out[offset + 66] = feat[offset + 66] * W_PALM_DIR
        out[offset + 67] = feat[offset + 67] * W_PALM_DIR
        out[offset + 63] = feat[offset + 63] * W_POSITION
        out[offset + 64] = feat[offset + 64] * W_POSITION
        out[offset + 65] = feat[offset + 65] * W_POSITION
    out[136:146] = feat[136:146] * W_ANCHOR
    out[149:161] = feat[149:161] * W_ARM         # 手臂特徵
    out[161:167] = feat[161:167] * W_FACE        # 手臉距離
    out[167:169] = feat[167:169] * W_ORIENT      # 身體朝向
    out[169:179] = feat[169:179] * W_HANDSHAPE   # 手指伸展分數
    return out

# Pose landmark 索引
NOSE         = 0
MOUTH_L      = 9
MOUTH_R      = 10
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

LABEL_ALIASES = {
    "現在": "現在／今天",
    "安靜": "安靜／平安",
    "相見": "相見／見面",
}

CONF_THRESHOLD      = 0.6   # 正常路徑信心門檻（滿幀 or 停止觸發）
EARLY_TRIGGER_CONF  = 0.75  # 提早觸發信心門檻
MIN_FRAMES          = 12    # 開始預覽推論的最少幀數
PREVIEW_INTERVAL    = 4     # 每隔幾幀跑一次預覽推論（不是每幀，節省算力）
CONSISTENCY_REQUIRED = 2    # 提早觸發需連續幾次相同預測（防止吸引子誤判）
SENTENCE_PAUSE      = 4.0
NO_HAND_TOLERANCE   = 5
STOP_THRESHOLD      = 0.01
STOP_FRAMES         = 5
RESULT_SHOW_SEC     = 2.5

CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17)
]

# ────────────────────────────────────────────────────────────
#  下載 MediaPipe 模型
# ────────────────────────────────────────────────────────────
import urllib.request
def _dl(path, url):
    if not os.path.exists(path):
        print(f"下載 {path} ...")
        urllib.request.urlretrieve(url, path)
        print(f"✅ {path}")

_dl(HAND_MODEL, "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
_dl(POSE_MODEL, "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task")

# ────────────────────────────────────────────────────────────
#  初始化 MediaPipe（Hand + Pose）
# ────────────────────────────────────────────────────────────
detector = vision.HandLandmarker.create_from_options(
    vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=HAND_MODEL),
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
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
print("MediaPipe 初始化完成（Hand + Pose）")

# ────────────────────────────────────────────────────────────
#  載入 BiLSTM
# ────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"裝置：{device}")

with open(CONFIG_PATH, "rb") as f:
    config = pickle.load(f)
with open(LABEL_PATH, "rb") as f:
    le = pickle.load(f)

model = SignLanguageBiLSTM(
    input_dim=config["input_dim"],
    num_classes=config["num_classes"],
    hidden_dim=config["hidden_dim"],
    num_layers=config["num_layers"],
    dropout=config["dropout"],
).to(device)

ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

feat_mean = torch.FloatTensor(config["feat_mean"]).to(device)
feat_std  = torch.FloatTensor(config["feat_std"]).to(device)
SEQ_LEN   = config["seq_len"]
print(f"BiLSTM 載入完成（Val Acc: {ckpt['val_acc']*100:.2f}%）")

# ────────────────────────────────────────────────────────────
#  中文字型
# ────────────────────────────────────────────────────────────
FONT_PATH = None
for p in ["C:/Windows/Fonts/msjh.ttc", "C:/Windows/Fonts/mingliu.ttc", "C:/Windows/Fonts/kaiu.ttf"]:
    if os.path.exists(p):
        FONT_PATH = p
        break

def put_chinese(img_bgr, text, pos, size=24, color=(255, 255, 255)):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.truetype(FONT_PATH, size) if FONT_PATH else ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()
    draw.text(pos, text, font=font, fill=color)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

# ────────────────────────────────────────────────────────────
#  Qwen（背景執行緒）
# ────────────────────────────────────────────────────────────
qwen_model, qwen_tokenizer = None, None

def load_qwen():
    global qwen_model, qwen_tokenizer
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        print("載入 Qwen2.5-7B（背景）...")
        bnb = BitsAndBytesConfig(load_in_4bit=True)
        qwen_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
        qwen_model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-7B-Instruct", quantization_config=bnb,
            device_map="auto", trust_remote_code=True)
        print("Qwen 載入完成")
    except Exception as e:
        print(f"Qwen 未載入（{e}）")

def qwen_correct(words):
    if qwen_model is None:
        return "、".join(words)
    raw = "、".join(words)
    prompt = f"以下是台灣手語辨識詞彙：「{raw}」\n請組合成通順的繁體中文句子，只輸出句子。"
    msgs = [{"role": "user", "content": prompt}]
    text = qwen_tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = qwen_tokenizer([text], return_tensors="pt").to(qwen_model.device)
    with torch.no_grad():
        out = qwen_model.generate(**inputs, max_new_tokens=64, do_sample=False)
    return qwen_tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

def send_to_qwen(words, sentence_log):
    print(f"\n送出詞彙：{'、'.join(words)}")
    result = qwen_correct(words)
    sentence_log.append(result)
    print(f"Qwen 輸出：{result}\n")

# ────────────────────────────────────────────────────────────
#  BiLSTM 推論
# ────────────────────────────────────────────────────────────
@torch.no_grad()
def predict_frames(frames_list):
    """
    接受變長的幀列表，均勻重採樣到 SEQ_LEN 後推論。
    frames_list: list of np.ndarray, 每個 shape (FEAT_DIM,)
    """
    seq = np.array(frames_list, dtype=np.float32)
    T = len(seq)
    if T != SEQ_LEN:
        indices = np.linspace(0, T - 1, SEQ_LEN, dtype=int)
        seq = seq[indices]
    x = torch.FloatTensor(seq).unsqueeze(0).to(device)
    x = (x - feat_mean) / feat_std
    logits = model(x)
    probs = torch.softmax(logits, dim=1)[0]
    top3_conf, top3_idx = probs.topk(3)
    top3 = [(LABEL_ALIASES.get(le.inverse_transform([i.item()])[0],
                              le.inverse_transform([i.item()])[0]), c.item())
            for i, c in zip(top3_idx, top3_conf)]
    return top3[0][0], top3[0][1], top3

# ────────────────────────────────────────────────────────────
#  主程式
# ────────────────────────────────────────────────────────────
print("\n開啟攝影機...")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("無法開啟攝影機")
    exit()

for _ in range(5):
    ret, _ = cap.read()
    if ret:
        break

print("攝影機已就緒，開始辨識...")
# threading.Thread(target=load_qwen, daemon=True).start()  # Qwen 暫停用

# ── 狀態機：IDLE → COLLECTING → PREDICTING → IDLE ──
seg_state      = "IDLE"
sign_frames    = []
no_hand_count  = 0
stop_frames    = 0
prev_wrist     = None

# 預覽推論（每 PREVIEW_INTERVAL 幀更新一次）
preview_top3     = []   # 即時候選，顯示在畫面上但尚未提交
consistent_label = ""   # 連續出現相同預測的標籤
consistent_count = 0    # 連續次數（達到 CONSISTENCY_REQUIRED 才提早提交）

word_buffer    = []
sentence_log   = []
last_hand_time = time.time()
last_pred      = None
paused         = False

# 顯示用
last_top3      = []
result_until   = 0.0

print("辨識中（Q=退出  C=清空  S=送出  空白=暫停）")
print("-" * 50)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    now = time.time()

    # ── MediaPipe 偵測（Hand + Pose）──
    rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result      = detector.detect(mp_image)
    pose_result = pose_detector.detect(mp_image)

    hand_detected = bool(result.hand_landmarks)
    feat = None

    # ── 畫軀幹（不論有無偵測到手，保持常駐顯示）──
    if pose_result.pose_landmarks:
        _pose = pose_result.pose_landmarks[0]
        _pts  = {
            "nose":    _pose[NOSE],
            "l_sho":   _pose[L_SHOULDER],
            "r_sho":   _pose[R_SHOULDER],
            "l_elbow": _pose[L_ELBOW],
            "r_elbow": _pose[R_ELBOW],
            "l_hip":   _pose[L_HIP],
            "r_hip":   _pose[R_HIP],
        }
        for pt in _pts.values():
            cv2.circle(frame, (int(pt.x * w), int(pt.y * h)), 7, (255, 100, 0), -1)
        for a_key, b_key in [("l_sho", "r_sho"),
                              ("r_sho", "r_elbow"), ("l_sho", "l_elbow")]:
            a, b = _pts[a_key], _pts[b_key]
            cv2.line(frame,
                     (int(a.x * w), int(a.y * h)),
                     (int(b.x * w), int(b.y * h)),
                     (255, 100, 0), 2)

    if hand_detected:
        last_hand_time = now

        # 畫雙手骨架
        for hand in result.hand_landmarks:
            for lm in hand:
                cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 5, (0, 255, 0), -1)
            for a_idx, b_idx in CONNECTIONS:
                a, b = hand[a_idx], hand[b_idx]
                cv2.line(frame,
                         (int(a.x * w), int(a.y * h)),
                         (int(b.x * w), int(b.y * h)),
                         (0, 200, 255), 2)

        # 分左右手（flip 後 MediaPipe Left = 畫面右手）
        right_hand, left_hand = None, None
        for i, cls in enumerate(result.handedness):
            label = cls[0].category_name
            if label == "Left":
                right_hand = result.hand_landmarks[i]
            else:
                left_hand  = result.hand_landmarks[i]

        # Pose 錨點
        if pose_result.pose_landmarks:
            pose  = pose_result.pose_landmarks[0]
            nose  = np.array([pose[NOSE].x,       pose[NOSE].y],       dtype=np.float32)
            l_sho = np.array([pose[L_SHOULDER].x,  pose[L_SHOULDER].y], dtype=np.float32)
            r_sho = np.array([pose[R_SHOULDER].x,  pose[R_SHOULDER].y], dtype=np.float32)
            l_hip = np.array([pose[L_HIP].x,       pose[L_HIP].y],      dtype=np.float32)
            r_hip = np.array([pose[R_HIP].x,       pose[R_HIP].y],      dtype=np.float32)
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
            l_sho        = np.array([0.4, 0.5], dtype=np.float32)
            r_sho        = np.array([0.6, 0.5], dtype=np.float32)
            mid_sho      = np.array([0.5, 0.5], dtype=np.float32)
            sho_width    = 0.3
            body_anchors = np.zeros(10, dtype=np.float32)
            r_elbow_lm = l_elbow_lm = r_wrist_p_lm = l_wrist_p_lm = None
            r_sho_lm   = l_sho_lm  = None
            mouth_c  = np.zeros(2, dtype=np.float32)
            r_ear_pt = np.zeros(2, dtype=np.float32)
            l_ear_pt = np.zeros(2, dtype=np.float32)

        def hand_feat(hand):
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
            return np.concatenate([coords, rel_w, pd])

        def compute_arm_feat(elbow_lm, wrist_lm, sho_lm):
            """6 維：手肘相對位置(2) + 上臂方向(2) + 前臂方向(2)"""
            if elbow_lm is None or sho_lm is None:
                return np.zeros(6, dtype=np.float32)
            elbow = np.array([elbow_lm.x, elbow_lm.y], dtype=np.float32)
            sho   = np.array([sho_lm.x,   sho_lm.y],   dtype=np.float32)
            elbow_rel = np.array([(elbow[0] - mid_sho[0]) / sho_width,
                                   (elbow[1] - mid_sho[1]) / sho_width], dtype=np.float32)
            upper   = elbow - sho
            upper   = upper / (np.linalg.norm(upper) + 1e-6)
            if wrist_lm is not None:
                wrist   = np.array([wrist_lm.x, wrist_lm.y], dtype=np.float32)
                forearm = wrist - elbow
                forearm = forearm / (np.linalg.norm(forearm) + 1e-6)
            else:
                forearm = np.zeros(2, dtype=np.float32)
            return np.concatenate([elbow_rel, upper, forearm])

        def face_dist_feat(hand, ear_pt):
            """3 維：手腕到鼻子/嘴巴/耳朵的正規化距離"""
            if hand is None:
                return np.zeros(3, dtype=np.float32)
            wrist = np.array([hand[0].x, hand[0].y], dtype=np.float32)
            d_nose  = np.linalg.norm(wrist - nose)    / sho_width
            d_mouth = np.linalg.norm(wrist - mouth_c) / sho_width
            d_ear   = np.linalg.norm(wrist - ear_pt)  / sho_width
            return np.array([d_nose, d_mouth, d_ear], dtype=np.float32)

        # 身體朝向（從即時肩膀座標推導）
        sho_vec    = r_sho - l_sho
        theta      = np.arctan2(sho_vec[1], sho_vec[0])
        body_orient = np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)

        raw167 = np.concatenate([
            hand_feat(right_hand), hand_feat(left_hand),
            body_anchors, np.zeros(3, dtype=np.float32),
            compute_arm_feat(r_elbow_lm, r_wrist_p_lm, r_sho_lm),
            compute_arm_feat(l_elbow_lm, l_wrist_p_lm, l_sho_lm),
            face_dist_feat(right_hand, r_ear_pt),
            face_dist_feat(left_hand,  l_ear_pt),
        ])
        # 計算手指伸展分數（從原始座標衍生）
        ext_r = compute_finger_extensions(raw167[:63])    if right_hand else np.zeros(5, dtype=np.float32)
        ext_l = compute_finger_extensions(raw167[68:131]) if left_hand  else np.zeros(5, dtype=np.float32)
        feat  = np.concatenate([raw167, body_orient, ext_r, ext_l])    # 179 維
        feat  = apply_feature_weights(feat)

        # ── 計算手腕速度（相對肩寬正規化，排除身體整體移動）──
        dominant = right_hand if right_hand is not None else left_hand
        cur_wrist = np.array([
            (dominant[0].x - mid_sho[0]) / sho_width,
            (dominant[0].y - mid_sho[1]) / sho_width,
        ], dtype=np.float32)
        wrist_speed = float(np.linalg.norm(cur_wrist - prev_wrist)) if prev_wrist is not None else 1.0
        prev_wrist = cur_wrist

        if wrist_speed < STOP_THRESHOLD:
            stop_frames += 1
        else:
            stop_frames = 0

        # ── 狀態機 ──
        if not paused:
            no_hand_count = 0

            if seg_state == "IDLE":
                seg_state        = "COLLECTING"
                sign_frames      = [feat]
                preview_top3     = []
                consistent_label = ""
                consistent_count = 0

            elif seg_state == "COLLECTING":
                sign_frames.append(feat)
                triggered = False

                # ── 預覽推論（每 PREVIEW_INTERVAL 幀更新）──
                if len(sign_frames) >= MIN_FRAMES and len(sign_frames) % PREVIEW_INTERVAL == 0:
                    lbl, conf, top3 = predict_frames(sign_frames)
                    preview_top3 = top3

                    # 一致性計數（防止吸引子誤判）
                    if lbl == consistent_label:
                        consistent_count += 1
                    else:
                        consistent_label = lbl
                        consistent_count = 1

                    # 路徑一：提早觸發（連續 N 次相同預測 且 信心度夠高）
                    if consistent_count >= CONSISTENCY_REQUIRED and conf >= EARLY_TRIGGER_CONF:
                        last_top3    = top3
                        result_until = now + RESULT_SHOW_SEC
                        preview_top3 = []
                        if lbl != last_pred and lbl != "_unknown_":
                            word_buffer.append(lbl)
                            last_pred = lbl
                            print(f"  [{conf*100:.0f}%] {lbl}  ({len(sign_frames)}幀, 一致×{consistent_count} 提早觸發)")
                        elif lbl == "_unknown_":
                            print(f"  [unknown {conf*100:.0f}%] 辨識為負樣本，略過")
                        sign_frames      = []
                        stop_frames      = 0
                        consistent_label = ""
                        consistent_count = 0
                        seg_state        = "IDLE"
                        triggered        = True

                # 路徑二：手腕停止 OR 窗口滿（正常門檻）
                if not triggered and (
                    len(sign_frames) >= SEQ_LEN or
                    (stop_frames >= STOP_FRAMES and len(sign_frames) >= MIN_FRAMES)
                ):
                    lbl, conf, top3 = predict_frames(sign_frames)
                    last_top3    = top3
                    result_until = now + RESULT_SHOW_SEC
                    preview_top3 = []
                    if conf >= CONF_THRESHOLD and lbl != last_pred and lbl != "_unknown_":
                        word_buffer.append(lbl)
                        last_pred = lbl
                        trigger = "停止" if stop_frames >= STOP_FRAMES else "滿幀"
                        print(f"  [{conf*100:.0f}%] {lbl}  ({len(sign_frames)}幀, {trigger}觸發)")
                    elif lbl == "_unknown_":
                        print(f"  [unknown {conf*100:.0f}%] 辨識為負樣本，略過")
                    else:
                        print(f"  [skip {conf*100:.0f}%] {lbl}  ({len(sign_frames)}幀)")
                    sign_frames      = []
                    stop_frames      = 0
                    consistent_label = ""
                    consistent_count = 0
                    seg_state        = "IDLE"

    if not hand_detected and not paused:
        no_hand_count += 1
        prev_wrist  = None   # 手消失，下次出現重新計算速度
        stop_frames = 0
        if seg_state == "COLLECTING" and no_hand_count > NO_HAND_TOLERANCE:
            seg_state        = "IDLE"
            sign_frames      = []
            preview_top3     = []
            consistent_label = ""
            consistent_count = 0
            no_hand_count    = 0

    # ── 停頓 → 清空（Qwen 暫停用，改為直接保留詞彙）──
    if (now - last_hand_time) > SENTENCE_PAUSE and word_buffer:
        pass  # 不自動清空，讓使用者手動按 C 或 S

    # ── UI ──
    if paused:
        status_text  = "[ 暫停 ]"
        status_color = (100, 100, 100)
    elif seg_state == "COLLECTING":
        status_text  = f"收集中  {len(sign_frames)}/{SEQ_LEN} 幀"
        status_color = (0, 80, 255)
        cv2.circle(frame, (620, 20), 10, (0, 0, 255), -1)
    elif hand_detected:
        status_text  = "偵測到手"
        status_color = (0, 200, 120)
    else:
        status_text  = "等待手勢"
        status_color = (120, 120, 120)

    frame = put_chinese(frame, status_text, (10, 8), size=22, color=status_color)

    words_str = "  ".join(word_buffer[-5:]) if word_buffer else "（空）"
    frame = put_chinese(frame, f"詞彙：{words_str}", (10, 40), size=20, color=(255, 255, 100))

    # 推論結果（顯示 RESULT_SHOW_SEC 秒）
    if now < result_until and last_top3:
        for i, (lbl, conf) in enumerate(last_top3):
            color = (200, 255, 200) if i == 0 else (150, 150, 150)
            frame = put_chinese(frame, f"{i+1}. {lbl}  {conf*100:.0f}%",
                                (10, 108 + i * 28), size=19, color=color)
    elif seg_state == "COLLECTING":
        # 進度條
        prog = int(len(sign_frames) / SEQ_LEN * 200)
        cv2.rectangle(frame, (10, 108), (10 + prog, 118), (0, 80, 255), -1)
        cv2.rectangle(frame, (10, 108), (210, 118), (80, 80, 80), 1)
        # 即時預覽（淡色，尚未提交）
        if preview_top3:
            for i, (lbl, conf) in enumerate(preview_top3):
                color = (120, 160, 120) if i == 0 else (90, 90, 90)
                frame = put_chinese(frame, f"{i+1}. {lbl}  {conf*100:.0f}%",
                                    (10, 122 + i * 26), size=18, color=color)

    frame = put_chinese(frame, "Q:退出  C:清空  空白:暫停",
                        (10, 455), size=17, color=(160, 160, 160))

    cv2.imshow("Taiwan Sign Language", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        word_buffer.clear()
        sign_frames   = []
        last_top3     = []
        result_until  = 0.0
        last_pred     = None
        seg_state        = "IDLE"
        no_hand_count    = 0
        stop_frames      = 0
        prev_wrist       = None
        preview_top3     = []
        consistent_label = ""
        consistent_count = 0
        print("  已清空")
    elif key == ord(' '):
        paused = not paused
        print(f"  {'暫停' if paused else '繼續'}")

cap.release()
cv2.destroyAllWindows()

if sentence_log:
    print("\n本次辨識紀錄：")
    for i, s in enumerate(sentence_log, 1):
        print(f"  {i}. {s}")
