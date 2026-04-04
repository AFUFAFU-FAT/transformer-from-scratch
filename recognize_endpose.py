"""
recognize_endpose.py（實驗版）
複製 recognize.py 的完整 BiLSTM 邏輯，
在「手停止觸發」路徑額外加入結尾單幀預測，合併兩者機率。

差異：
  - 手停止時：sequence_probs * SEQ_W + endpose_probs * END_W
  - 若兩者 top1 一致 → 信心度更穩
  - 若不一致     → 以序列為主，降低信心避免誤判
  - 畫面顯示兩種預測結果，方便比對
"""

import cv2, math
import lm_selector
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import torch
import torch.nn as nn
import pickle, time, os, csv, datetime
from collections import deque
from PIL import Image, ImageDraw, ImageFont
import urllib.request

# ────────────────────────────────────────────────────────────
#  模型（與 train_lstm.py 完全一致）
# ────────────────────────────────────────────────────────────
class SignLanguageBiLSTM(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=64, num_layers=1, dropout=0.6):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, 64), nn.LayerNorm(64), nn.ReLU(),
        )
        self.lstm = nn.LSTM(
            input_size=64, hidden_size=hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0, bidirectional=True,
        )
        self.attention  = nn.Linear(hidden_dim * 2, 1)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim * 2, num_classes))
        self.dropout    = nn.Dropout(dropout)

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
MODEL_PATH  = "models/lstm_best.pth"
CONFIG_PATH = "models/lstm_config.pkl"
LABEL_PATH  = "data/seq_label_encoder.pkl"
HAND_MODEL  = "hand_landmarker.task"
POSE_MODEL  = "pose_landmarker_lite.task"

FEAT_DIM    = 199   # 167 + 身體朝向2 + 伸展分數10 + 伸展二值10 + 指向方向6 + 捏合距離2 + 手部旗標2

W_HANDSHAPE = 6.0; W_FINGERTIP = 4.0; W_FINGER = 3.0; W_BINARY = 10.0
W_PALM_DIR  = 3.0; W_POSITION  = 4.0; W_ARM    = 2.5; W_POINTING = 8.0
LOG_PATH = "data/test_log.csv"
W_FACE = 5.0; W_ORIENT = 2.0; W_ANCHOR = 1.5
FINGERTIPS = {4, 8, 12, 16, 20}
TIP_IDX = [4, 8, 12, 16, 20]; MCP_IDX = [1, 5, 9, 13, 17]
FINGER_JOINTS = [(2,3,4),(5,6,7),(9,10,11),(13,14,15),(17,18,19)]
EXTEND_ANGLE  = 140.0

NOSE = 0; MOUTH_L = 9; MOUTH_R = 10; L_EAR = 7; R_EAR = 8
L_SHOULDER=11; R_SHOULDER=12; L_ELBOW=13; R_ELBOW=14
L_WRIST_POSE=15; R_WRIST_POSE=16; L_HIP=23; R_HIP=24

CONF_THRESHOLD         = 0.25  # 降低門檻，讓「我」等低幅度詞彙能輸出
LM_CONF_THRESHOLD      = 0.70  # 低於此值且候選接近時才呼叫 LLM
CANDIDATE_OFFSET       = 4     # 候選切法偏移幀數
RAMP_FRAMES            = 6      # 每段序列開頭丟棄的過渡幀數
EARLY_TRIGGER_CONF     = 0.90
EARLY_ENDPOSE_CONF     = 0.82  # endpose 信心度達此值 → 不等滿 STOP_FRAMES
EARLY_ENDPOSE_MIN_STABLE = 3   # 至少靜止幾幀才做 endpose 提早檢查
MIN_FRAMES           = 12
PREVIEW_INTERVAL     = 4
CONSISTENCY_REQUIRED = 3
SENTENCE_PAUSE       = 4.0
NO_HAND_TOLERANCE    = 5
STOP_THRESHOLD       = 0.01
STOP_FRAMES          = 8
RESULT_SHOW_SEC      = 2.5

# ── 結尾合併設定 ──
SEQ_W          = 0.5    # 序列預測權重
END_W          = 0.5    # 結尾姿勢預測權重
STABLE_BUFFER  = 8      # ring buffer 大小（存最近幾幀做結尾平均）

LABEL_ALIASES = {"現在": "現在／今天", "安靜": "安靜／平安", "相見": "相見／見面"}

CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17)
]

# ────────────────────────────────────────────────────────────
#  特徵工程
# ────────────────────────────────────────────────────────────
def compute_finger_extensions(lm63):
    lms = lm63.reshape(21, 3); wrist = lms[0]
    scores = np.empty(5, dtype=np.float32)
    for fi, (ti, mi) in enumerate(zip(TIP_IDX, MCP_IDX)):
        scores[fi] = np.linalg.norm(lms[ti]-wrist) / (np.linalg.norm(lms[mi]-wrist)+1e-6)
    return scores

def compute_pointing_dir(lm63):
    """食指指向方向（3D 正規化向量）：normalize(tip[8] - MCP[5])"""
    lms = lm63.reshape(21, 3)
    direction = lms[8] - lms[5]
    norm = np.linalg.norm(direction) + 1e-6
    return (direction / norm).astype(np.float32)

def compute_finger_binary(lm63):
    """5-dim: 1=伸出, 0=彎曲（純 PIP 關節角度，無遮蔽估計）"""
    lms = lm63.reshape(21, 3)
    binary = np.empty(5, dtype=np.float32)
    for fi, (a, b, c) in enumerate(FINGER_JOINTS):
        v1 = lms[a] - lms[b]; v2 = lms[c] - lms[b]
        cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        binary[fi] = 1.0 if np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))) > EXTEND_ANGLE else 0.0
    return binary

def compute_pinch_dist(lm63):
    """捏合距離：||拇指尖-食指尖|| / 掌長。找(OK)≈0，再見/認真≈大"""
    lms = lm63.reshape(21, 3)
    dist = np.linalg.norm(lms[4] - lms[8]) / (np.linalg.norm(lms[9] - lms[0]) + 1e-6)
    return np.array([min(dist, 3.0)], dtype=np.float32)

def apply_feature_weights(feat):
    out = feat.copy()
    for offset in (0, 68):
        wx, wy = feat[offset], feat[offset+1]
        mx, my = feat[offset+27], feat[offset+28]
        scale  = np.sqrt((mx-wx)**2+(my-wy)**2)+1e-6
        for i in range(21):
            b = offset+i*3; w = W_FINGERTIP if i in FINGERTIPS else W_FINGER
            out[b]=(feat[b]-wx)/scale*w; out[b+1]=(feat[b+1]-wy)/scale*w; out[b+2]=feat[b+2]*w
        out[offset+66]=feat[offset+66]*W_PALM_DIR; out[offset+67]=feat[offset+67]*W_PALM_DIR
        out[offset+63]=feat[offset+63]*W_POSITION; out[offset+64]=feat[offset+64]*W_POSITION
        out[offset+65]=feat[offset+65]*W_POSITION
    out[136:146]=feat[136:146]*W_ANCHOR
    out[149:161]=feat[149:161]*W_ARM
    out[161:167]=feat[161:167]*W_FACE
    out[167:169]=feat[167:169]*W_ORIENT
    out[169:179]=feat[169:179]*W_HANDSHAPE
    out[179:189]=feat[179:189]*W_BINARY       # 手指伸展二值
    out[189:195]=feat[189:195]*W_POINTING    # 食指指向方向
    out[195:197]=feat[195:197]*W_BINARY      # 捏合距離
    out[197:199]=feat[197:199]*W_BINARY      # 手部偵測旗標
    return out

# ────────────────────────────────────────────────────────────
#  字型
# ────────────────────────────────────────────────────────────
FONT_PATH = None
for p in ["C:/Windows/Fonts/msjh.ttc","C:/Windows/Fonts/mingliu.ttc","C:/Windows/Fonts/kaiu.ttf"]:
    if os.path.exists(p): FONT_PATH = p; break

def put_chinese(img, text, pos, size=24, color=(255,255,255)):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb); draw = ImageDraw.Draw(pil)
    try:    font = ImageFont.truetype(FONT_PATH, size) if FONT_PATH else ImageFont.load_default()
    except: font = ImageFont.load_default()
    draw.text(pos, text, font=font, fill=color)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

# ────────────────────────────────────────────────────────────
#  下載 + 初始化 MediaPipe
# ────────────────────────────────────────────────────────────
def _dl(path, url):
    if not os.path.exists(path):
        print(f"下載 {path}..."); urllib.request.urlretrieve(url, path)

_dl(HAND_MODEL,"https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
_dl(POSE_MODEL,"https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task")

detector = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=HAND_MODEL),
    num_hands=2, min_hand_detection_confidence=0.4,
    min_hand_presence_confidence=0.4, min_tracking_confidence=0.4,
))
pose_detector = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=POSE_MODEL),
    min_pose_detection_confidence=0.4, min_pose_presence_confidence=0.4, min_tracking_confidence=0.4,
))
print("MediaPipe 初始化完成（Hand + Pose）")

# ────────────────────────────────────────────────────────────
#  載入 BiLSTM
# ────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"裝置：{device}")

with open(CONFIG_PATH,"rb") as f: config = pickle.load(f)
with open(LABEL_PATH, "rb") as f: le     = pickle.load(f)

model = SignLanguageBiLSTM(
    input_dim=config["input_dim"], num_classes=config["num_classes"],
    hidden_dim=config["hidden_dim"], num_layers=config["num_layers"], dropout=config["dropout"],
).to(device)
ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

feat_mean = torch.FloatTensor(config["feat_mean"]).to(device)
feat_std  = torch.FloatTensor(config["feat_std"]).to(device)
SEQ_LEN   = config["seq_len"]
print(f"BiLSTM 載入完成（Val Acc: {ckpt['val_acc']*100:.2f}%）")

# ────────────────────────────────────────────────────────────
#  推論函數
# ────────────────────────────────────────────────────────────
@torch.no_grad()
def get_probs(frames_list):
    """frames_list → softmax 機率向量 (num_classes,)"""
    seq = np.array(frames_list, dtype=np.float32)
    T = len(seq)
    if T != SEQ_LEN:
        indices = np.linspace(0, T-1, SEQ_LEN, dtype=int)
        seq = seq[indices]
    # 若尚未加 cumulative（398-dim），自動補上
    if seq.shape[1] == 398:
        right_start = seq[0:1, 63:66]
        left_start  = seq[0:1, 131:134]
        cum_r = seq[:, 63:66]   - right_start
        cum_l = seq[:, 131:134] - left_start
        seq = np.concatenate([seq, cum_r, cum_l], axis=1)
    x = (torch.FloatTensor(seq).unsqueeze(0).to(device) - feat_mean) / feat_std
    return torch.softmax(model(x), dim=1)[0]

def probs_to_top3(probs):
    top3_conf, top3_idx = probs.topk(3)
    return [(LABEL_ALIASES.get(le.classes_[i.item()], le.classes_[i.item()]), c.item())
            for i, c in zip(top3_idx, top3_conf)]

def add_cumulative(frames_arr):
    """frames_arr: (T, D) → (T, D+6)，追加右/左手腕累積位移"""
    right_start = frames_arr[0:1, 63:66]
    left_start  = frames_arr[0:1, 131:134]
    cum_r = frames_arr[:, 63:66]   - right_start
    cum_l = frames_arr[:, 131:134] - left_start
    return np.concatenate([frames_arr, cum_r, cum_l], axis=1)

def apply_feature_gates(probs, gate_feat_199, frames_398=None):
    """
    gate_feat_199: (199,) weighted base features（×W）
    frames_398:    list/array of 398-dim，用於計算手腕動作幅度
    若必要條件不滿足，該詞彙機率 × 0.05，再重新正規化。
    """
    bin_r   = gate_feat_199[179:184]   # ×10，0=彎 10=伸
    pinch_r = float(gate_feat_199[195])
    is_left = float(gate_feat_199[198])

    # 手腕動作幅度（右手 rel_wrist ×2.5，索引 63=X 64=Y）
    wrist_x_std = wrist_y_std = 0.0
    if frames_398 is not None and len(frames_398) > 2:
        farr = np.array(frames_398)
        wrist_x_std = float(np.std(farr[:, 63]))
        wrist_y_std = float(np.std(farr[:, 64]))

    GATES = {
        # ── 手形 gate ──
        # 找 = 中無小指伸出，食指明顯彎曲（拇指 IP 角度不可靠，不納入）
        # bin_r[1]=食指(dim180), bin_r[2]=中(181), bin_r[3]=無(182), bin_r[4]=小(183)
        '找':   (float(bin_r[1]) < 3.0 and           # 食指明顯彎曲（加權<3 = raw<0.3）
                 float(bin_r[2]) > 7.0 and           # 中指伸出
                 float(bin_r[3]) > 7.0 and           # 無名指伸出
                 float(bin_r[4]) > 7.0),             # 小拇指伸出
        '認真': is_left             >  5.0,          # 雙手，左手必須存在
        '甚麼': float(bin_r[0])     >  3.0,          # 拇指必須伸出
        '他':   float(bin_r[0])     <  6.0,          # 拇指低→壓制他（甚麼拇指高）
        '媽媽': float(bin_r[4])     >  3.0,          # 小指必須伸出
        # ── 動作 gate（正向）：動作型手勢 → 要有足夠手腕位移 ──
        '再見': wrist_x_std         >  0.06,         # 左右搖動，X 標準差夠大
        '喜歡': (wrist_x_std > 0.05 or wrist_y_std > 0.05),  # 上下/左右動作
        # ── 靜止 gate（反向）：靜止型手勢 → 有動作時壓制 ──
        # 當用戶做 再見(高x動作) 時壓制「是」；做 喜歡(高y動作) 時壓制「好」
        '是':   (wrist_x_std < 0.12 and wrist_y_std < 0.12),  # 是 = 靜止於下巴
        '好':   (wrist_x_std < 0.10 and wrist_y_std < 0.10),  # 好 = 靜止於臉側
    }
    p = probs.cpu().numpy().copy()
    for word, cond in GATES.items():
        idx_arr = np.where(le.classes_ == word)[0]
        if len(idx_arr) == 0: continue
        if not cond:
            p[idx_arr[0]] *= 0.05
    s = p.sum()
    if s > 1e-8: p /= s
    return torch.FloatTensor(p).to(probs.device)

def _gate_feat(frames_arr_398):
    """從 398-dim 序列取前 199 維的平均，作為 gate 用特徵"""
    return np.mean(np.array(frames_arr_398)[:, :199], axis=0)

def smooth_frames(frames_list):
    """
    對累積的 sign_frames（398-dim）施加 3 幀滑動平均，
    平滑 binary(179:189) 和 pointing(189:195) 特徵，去除跳幀噪音。
    訓練時 build_sequences.py 也做同樣的平滑，保持一致性。
    """
    arr = np.array(frames_list, dtype=np.float32)
    if len(arr) < 3:
        return frames_list
    result = arr.copy()
    kernel = np.array([1/3, 1/3, 1/3])
    # binary（已加權 ×10）和 pointing（已加權 ×8）平滑
    for i in range(179, 195):
        result[:, i] = np.convolve(result[:, i], kernel, mode='same')
    # pointing 方向重新正規化（乘回 W_POINTING=8）
    for start in (189, 192):
        chunk = result[:, start:start+3]
        norms = np.linalg.norm(chunk, axis=1, keepdims=True)
        result[:, start:start+3] = chunk / (norms + 1e-6) * 8.0
    return result.tolist()

def predict_frames(frames_list):
    arr   = add_cumulative(np.array(frames_list, dtype=np.float32))  # → 404-dim
    probs = get_probs(arr)
    probs = apply_feature_gates(probs, _gate_feat(arr), arr)        # ← 手形+動作 gate
    top3  = probs_to_top3(probs)
    return top3[0][0], top3[0][1], top3

def generate_candidates(frames):
    candidates = []
    frames = frames[RAMP_FRAMES:] if len(frames) > RAMP_FRAMES + MIN_FRAMES else frames
    frames = list(frames)
    n = len(frames)

    lbl, conf, top3 = predict_frames(frames)
    candidates.append({"key": "A", "label": lbl, "conf": conf, "top3": top3})

    if n - CANDIDATE_OFFSET >= MIN_FRAMES:
        lbl, conf, top3 = predict_frames(frames[:n - CANDIDATE_OFFSET])
        candidates.append({"key": "B", "label": lbl, "conf": conf, "top3": top3})

    if n - CANDIDATE_OFFSET >= MIN_FRAMES:
        lbl, conf, top3 = predict_frames(frames[CANDIDATE_OFFSET:])
        candidates.append({"key": "C", "label": lbl, "conf": conf, "top3": top3})

    return candidates

def select_from_candidates(candidates, word_buffer):
    """
    信心度夠高 → 直接選最佳；否則呼叫 LLM 評分。
    回傳 (final_label, final_conf, method_note)
    """
    best = max(candidates, key=lambda c: c["conf"])
    if best["conf"] >= LM_CONF_THRESHOLD:
        return best["label"], best["conf"], f"候選{best['key']}"
    lbl = lm_selector.lm_select(candidates, word_buffer)
    conf = next((c["conf"] for c in candidates if c["label"] == lbl), best["conf"])
    return lbl, conf, "LM評分"

@torch.no_grad()
def predict_with_endpose(sign_frames, stable_buffer):
    """
    合併：sequence_probs * SEQ_W + endpose_probs * END_W
    stable_buffer：deque of (FEAT_DIM,) 最近幾幀靜止特徵
    """
    seq_probs = get_probs(sign_frames)

    # 結尾姿勢：取 stable_buffer 的平均幀，重複 SEQ_LEN 次（靜止→cumulative=0）
    avg_feat_base = np.mean(np.array(stable_buffer), axis=0)
    avg_feat  = np.concatenate([avg_feat_base, np.zeros(6, dtype=np.float32)])
    end_probs = get_probs([avg_feat] * SEQ_LEN)

    # ── 自適應 endpose 權重 ────────────────────────────────────
    # 若結尾姿勢信心不足（< 0.30），表示手勢結尾不固定（如：搖手、上下移動），
    # 降低 end_probs 佔比，以序列為主，避免不穩定 endpose 拉低整體信心
    end_top1_conf = float(end_probs.max())
    if end_top1_conf >= 0.30:
        eff_seq_w, eff_end_w = SEQ_W, END_W
    else:
        eff_seq_w, eff_end_w = 0.75, 0.25  # endpose 不可靠 → 序列主導
    combined  = seq_probs * eff_seq_w + end_probs * eff_end_w

    # ── 手形＋動作 gate（序列）─────────────────────────────────
    sign_arr_404 = add_cumulative(np.array(sign_frames, dtype=np.float32))
    combined = apply_feature_gates(combined, _gate_feat(sign_arr_404), sign_arr_404)

    # ── 分類後處理：規則 gate ──────────────────────────────────
    buf_arr = np.array(stable_buffer)   # (N, 199) 已加權特徵

    # Gate 1：雙手 gate
    # 左手幀比例 < 40% → 視為「無左手」，排除穩定雙手詞彙
    left_rate = (buf_arr[:, 198] > W_BINARY * 0.5).mean()
    if left_rate < 0.4:
        BILATERAL_ONLY = {"對不起", "認真", "人", "聽人", "晚上"}
        p = combined.cpu().numpy().copy()
        for word in BILATERAL_ONLY:
            idx_arr = np.where(le.classes_ == word)[0]
            if len(idx_arr): p[idx_arr[0]] *= 0.02
        p /= (p.sum() + 1e-9)
        combined = torch.FloatTensor(p).to(combined.device)

    # Gate 2：說 的 z 方向 gate
    # 說 z_ptr≈-0.273 (weighted≈-2.18)，是≈-0.117 (-0.94)，閾值 -1.44
    # 若右手食指未明顯朝向鏡頭，壓制「說」
    point_z_w = float(avg_feat_base[191])        # dim 191 = right pointing z × W_POINTING
    if point_z_w > -0.18 * W_POINTING:           # 不夠朝向鏡頭
        p = combined.cpu().numpy().copy()
        idx_arr = np.where(le.classes_ == "說")[0]
        if len(idx_arr): p[idx_arr[0]] *= 0.1
        p /= (p.sum() + 1e-9)
        combined = torch.FloatTensor(p).to(combined.device)
    # ──────────────────────────────────────────────────────────

    top3_combined = probs_to_top3(combined)
    top3_seq      = probs_to_top3(seq_probs)
    top3_end      = probs_to_top3(end_probs)

    lbl, conf = top3_combined[0]
    return lbl, conf, top3_combined, top3_seq, top3_end

# ────────────────────────────────────────────────────────────
#  Post-hoc 規則 Gate（兩條觸發路徑共用）
# ────────────────────────────────────────────────────────────
_BILATERAL_ONLY = {"對不起", "認真", "人", "聽人", "晚上"}

def apply_gates_posthoc(final_lbl, final_conf, method, candidates, frame_src):
    """Override label based on hand-feature rules after candidate selection.
    Returns (final_lbl, final_conf, method, avg_frame).
    """
    if len(frame_src) == 0:
        return final_lbl, final_conf, method, None

    avg_frame  = frame_src.mean(axis=0)           # (FEAT_DIM,) 加權平均特徵
    left_rate  = (frame_src[:, 198] > W_BINARY * 0.5).mean()

    # ── Gate 1：無左手 → 不可能是雙手詞彙 ─────────────────────
    if left_rate < 0.4 and final_lbl in _BILATERAL_ONLY:
        for cand in sorted(candidates, key=lambda c: c["conf"], reverse=True):
            if cand["label"] not in _BILATERAL_ONLY:
                final_lbl, final_conf = cand["label"], cand["conf"]
                method += "+BiGate"
                break

    # ── Gate 3：說 的手指數量 gate（4+ 指伸出 → 不是說）────────
    # 注意：拇指 IP 角度方法不可靠（幾乎恆=1），故 threshold 用 179:184（5指）
    # 實際有效判斷只有食中無小（180:184），但整體 >= 4 足以排除「說」
    if final_lbl == "說":
        finger_count = int((avg_frame[179:184] > W_BINARY * 0.5).sum())
        if finger_count >= 4:
            for cand in sorted(candidates, key=lambda c: c["conf"], reverse=True):
                if cand["label"] != "說":
                    final_lbl, final_conf = cand["label"], cand["conf"]
                    method += "+FingGate"
                    break

    # ── Gate 4：找 的手形 gate ────────────────────────────────
    # 找 = 中指無名指小拇指伸出，食指彎曲，無左手
    # 拇指 IP 角度不可靠（恆≈1），故不納入判斷
    # 食指 binary（dim 180）在 找 中 avg=0.12 → 加權≈1.2 < W_BINARY*0.25=2.5
    # 中無小 binary（dim 181-183）≈ 1.0 → 加權 10 > W_BINARY*0.7=7
    if final_lbl != "找":
        index_bent        = avg_frame[180] < W_BINARY * 0.25          # 食指明顯彎曲
        mid_ring_pink_ext = (avg_frame[181:184] > W_BINARY * 0.7).all() # 中無小明顯伸出
        no_left_hand      = left_rate < 0.15                           # 嚴格無左手
        if index_bent and mid_ring_pink_ext and no_left_hand:
            for cand in sorted(candidates, key=lambda c: c["conf"], reverse=True):
                if cand["label"] == "找":
                    final_lbl, final_conf = cand["label"], cand["conf"]
                    method += "+ZhaoGate"
                    break

    return final_lbl, final_conf, method, avg_frame

# ────────────────────────────────────────────────────────────
#  測試記錄（test_log.csv）
# ────────────────────────────────────────────────────────────
_LOG_FIELDS = ["timestamp", "label", "conf", "method", "top1", "top2", "top3", "verified"]

def _write_log(row):
    file_exists = os.path.isfile(LOG_PATH)
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_LOG_FIELDS)
        if not file_exists:
            w.writeheader()
        w.writerow(row)

def log_prediction(label, conf, method, top3_list):
    """Record a prediction. Returns the pending row for later verification."""
    row = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "label": label,
        "conf": f"{conf*100:.1f}",
        "method": method,
        "top1": f"{top3_list[0][0]}:{top3_list[0][1]*100:.0f}" if len(top3_list) > 0 else "",
        "top2": f"{top3_list[1][0]}:{top3_list[1][1]*100:.0f}" if len(top3_list) > 1 else "",
        "top3": f"{top3_list[2][0]}:{top3_list[2][1]*100:.0f}" if len(top3_list) > 2 else "",
        "verified": "",
    }
    return row

# ────────────────────────────────────────────────────────────
#  主程式
# ────────────────────────────────────────────────────────────
print("\n開啟攝影機...")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
if not cap.isOpened(): print("無法開啟攝影機"); exit()
for _ in range(5):
    ret, _ = cap.read()
    if ret: break

print("攝影機已就緒，開始辨識...")
print("辨識中（Q=退出  C=清空  空白=暫停  Y=正確 N=錯誤）")
print(f"測試記錄將存入：{LOG_PATH}")
print("-" * 50)

seg_state        = "IDLE"
sign_frames      = []
stable_buffer    = deque(maxlen=STABLE_BUFFER)   # ← 結尾 ring buffer
no_hand_count    = 0
stop_frames_cnt  = 0
prev_wrist       = None
prev_feat_delta  = None   # ← 前一幀特徵，用於計算 delta
preview_top3     = []
consistent_label = ""
consistent_count = 0
word_buffer      = []
last_top3        = []
last_seq_top3    = []   # ← 顯示用：序列預測
last_end_top3    = []   # ← 顯示用：結尾預測
result_until     = 0.0
last_pred        = None
pending_log      = None   # ← 最後一次預測，等待 y/n 驗證
paused           = False
last_hand_time   = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    now   = time.time()

    rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result      = detector.detect(mp_image)
    pose_result = pose_detector.detect(mp_image)

    hand_detected = bool(result.hand_landmarks)
    feat = None

    if hand_detected:
        last_hand_time = now

        for hand in result.hand_landmarks:
            for lm in hand:
                cv2.circle(frame,(int(lm.x*w),int(lm.y*h)),5,(0,255,0),-1)
            for a,b in CONNECTIONS:
                pa,pb = hand[a],hand[b]
                cv2.line(frame,(int(pa.x*w),int(pa.y*h)),(int(pb.x*w),int(pb.y*h)),(0,200,255),2)

        right_hand = left_hand = None
        for i, cls in enumerate(result.handedness):
            if cls[0].category_name == "Left": right_hand = result.hand_landmarks[i]
            else:                               left_hand  = result.hand_landmarks[i]

        if pose_result.pose_landmarks:
            pose  = pose_result.pose_landmarks[0]
            nose  = np.array([pose[NOSE].x,      pose[NOSE].y],      dtype=np.float32)
            l_sho = np.array([pose[L_SHOULDER].x, pose[L_SHOULDER].y], dtype=np.float32)
            r_sho = np.array([pose[R_SHOULDER].x, pose[R_SHOULDER].y], dtype=np.float32)
            l_hip = np.array([pose[L_HIP].x,      pose[L_HIP].y],      dtype=np.float32)
            r_hip = np.array([pose[R_HIP].x,      pose[R_HIP].y],      dtype=np.float32)
            mid_sho   = (l_sho+r_sho)/2.0
            sho_width = max(abs(r_sho[0]-l_sho[0]),1e-6)
            body_anchors = np.concatenate([nose,l_sho,r_sho,l_hip,r_hip]).astype(np.float32)
            r_elbow_lm=pose[R_ELBOW]; l_elbow_lm=pose[L_ELBOW]
            r_wrist_p_lm=pose[R_WRIST_POSE]; l_wrist_p_lm=pose[L_WRIST_POSE]
            r_sho_lm=pose[R_SHOULDER]; l_sho_lm=pose[L_SHOULDER]
            mouth_c  = np.array([(pose[MOUTH_L].x+pose[MOUTH_R].x)/2,
                                  (pose[MOUTH_L].y+pose[MOUTH_R].y)/2], dtype=np.float32)
            r_ear_pt = np.array([pose[R_EAR].x, pose[R_EAR].y], dtype=np.float32)
            l_ear_pt = np.array([pose[L_EAR].x, pose[L_EAR].y], dtype=np.float32)
        else:
            nose=np.array([0.5,0.3],dtype=np.float32); l_sho=np.array([0.4,0.5],dtype=np.float32)
            r_sho=np.array([0.6,0.5],dtype=np.float32); mid_sho=np.array([0.5,0.5],dtype=np.float32)
            sho_width=0.3; body_anchors=np.zeros(10,dtype=np.float32)
            r_elbow_lm=l_elbow_lm=r_wrist_p_lm=l_wrist_p_lm=None
            r_sho_lm=l_sho_lm=None
            mouth_c=r_ear_pt=l_ear_pt=np.zeros(2,dtype=np.float32)

        def hand_feat(hand):
            if hand is None: return np.zeros(68,dtype=np.float32)
            coords=np.array([[lm.x,lm.y,lm.z] for lm in hand],dtype=np.float32).flatten()
            wrist=np.array([hand[0].x,hand[0].y,hand[0].z],dtype=np.float32)
            rel_w=np.array([(wrist[0]-mid_sho[0])/sho_width,(wrist[1]-mid_sho[1])/sho_width,wrist[2]],dtype=np.float32)
            pd=np.array([hand[9].x-hand[0].x,hand[9].y-hand[0].y],dtype=np.float32)
            pd=pd/(np.linalg.norm(pd)+1e-6)
            return np.concatenate([coords,rel_w,pd])

        def compute_arm_feat(elbow_lm,wrist_lm,sho_lm):
            if elbow_lm is None or sho_lm is None: return np.zeros(6,dtype=np.float32)
            elbow=np.array([elbow_lm.x,elbow_lm.y],dtype=np.float32)
            sho=np.array([sho_lm.x,sho_lm.y],dtype=np.float32)
            elbow_rel=np.array([(elbow[0]-mid_sho[0])/sho_width,(elbow[1]-mid_sho[1])/sho_width],dtype=np.float32)
            upper=elbow-sho; upper=upper/(np.linalg.norm(upper)+1e-6)
            if wrist_lm is not None:
                wrist=np.array([wrist_lm.x,wrist_lm.y],dtype=np.float32)
                forearm=wrist-elbow; forearm=forearm/(np.linalg.norm(forearm)+1e-6)
            else: forearm=np.zeros(2,dtype=np.float32)
            return np.concatenate([elbow_rel,upper,forearm])

        def face_dist_feat(hand,ear_pt):
            if hand is None: return np.zeros(3,dtype=np.float32)
            wrist=np.array([hand[0].x,hand[0].y],dtype=np.float32)
            return np.array([np.linalg.norm(wrist-nose)/sho_width,
                             np.linalg.norm(wrist-mouth_c)/sho_width,
                             np.linalg.norm(wrist-ear_pt)/sho_width],dtype=np.float32)

        sho_vec=r_sho-l_sho; theta=np.arctan2(sho_vec[1],sho_vec[0])
        body_orient=np.array([np.cos(theta),np.sin(theta)],dtype=np.float32)

        raw167=np.concatenate([hand_feat(right_hand),hand_feat(left_hand),
            body_anchors,np.zeros(3,dtype=np.float32),
            compute_arm_feat(r_elbow_lm,r_wrist_p_lm,r_sho_lm),
            compute_arm_feat(l_elbow_lm,l_wrist_p_lm,l_sho_lm),
            face_dist_feat(right_hand,r_ear_pt),face_dist_feat(left_hand,l_ear_pt),
        ])
        ext_r=compute_finger_extensions(raw167[:63])    if right_hand else np.zeros(5,dtype=np.float32)
        ext_l=compute_finger_extensions(raw167[68:131]) if left_hand  else np.zeros(5,dtype=np.float32)
        bin_r  =compute_finger_binary(raw167[:63])        if right_hand else np.zeros(5,dtype=np.float32)
        bin_l  =compute_finger_binary(raw167[68:131])     if left_hand  else np.zeros(5,dtype=np.float32)
        ptr_r  =compute_pointing_dir(raw167[:63])          if right_hand else np.zeros(3,dtype=np.float32)
        ptr_l  =compute_pointing_dir(raw167[68:131])       if left_hand  else np.zeros(3,dtype=np.float32)
        pinch_r=compute_pinch_dist(raw167[:63])            if right_hand else np.array([3.0],dtype=np.float32)
        pinch_l=compute_pinch_dist(raw167[68:131])         if left_hand  else np.array([3.0],dtype=np.float32)
        # ── debug：顯示右手 binary（拇食中無小）
        # 注意：拇指 IP 角度法不可靠（內收時角度仍≈180°），故拇指以 ? 標示
        _labels = "拇食中無小"
        def _fmt(b, lbl, is_thumb=False):
            if is_thumb: return "拇?"    # 拇指 IP 角度對內收/外展無效，永遠標 ?
            return lbl if b > 0.5 else "✗"
        _r = "".join(_fmt(bin_r[i], _labels[i], i==0) for i in range(5))
        _l = "".join(_fmt(bin_l[i], _labels[i], i==0) for i in range(5))
        frame=put_chinese(frame,f"右:{_r} 左:{_l}",(10,430),size=16,color=(180,255,180))
        is_right = np.array([1.0 if right_hand else 0.0], dtype=np.float32)
        is_left  = np.array([1.0 if left_hand  else 0.0], dtype=np.float32)
        feat=np.concatenate([raw167,body_orient,ext_r,ext_l,bin_r,bin_l,ptr_r,ptr_l,pinch_r,pinch_l,is_right,is_left])
        feat=apply_feature_weights(feat)

        # ── Delta（速度）特徵：追加 feat[t]-feat[t-1]，靜止動作 delta≈0 ──
        delta = feat - prev_feat_delta if prev_feat_delta is not None else np.zeros_like(feat)
        prev_feat_delta = feat.copy()
        feat = np.concatenate([feat, delta])   # 197 → 394 維

        dominant=right_hand if right_hand else left_hand
        cur_wrist=np.array([(dominant[0].x-mid_sho[0])/sho_width,
                             (dominant[0].y-mid_sho[1])/sho_width],dtype=np.float32)
        wrist_speed=float(np.linalg.norm(cur_wrist-prev_wrist)) if prev_wrist is not None else 1.0
        prev_wrist=cur_wrist

        if wrist_speed < STOP_THRESHOLD:
            stop_frames_cnt += 1
            stable_buffer.append(feat.copy())   # ← 靜止時才加入結尾 buffer
        else:
            stop_frames_cnt = 0

        if not paused:
            no_hand_count = 0

            if seg_state == "IDLE":
                seg_state="COLLECTING"; sign_frames=[feat]
                preview_top3=[]; consistent_label=""; consistent_count=0

            elif seg_state == "COLLECTING":
                sign_frames.append(feat)
                triggered = False

                if len(sign_frames) >= MIN_FRAMES and len(sign_frames) % PREVIEW_INTERVAL == 0:
                    lbl, conf, top3 = predict_frames(sign_frames)
                    preview_top3 = top3
                    if lbl == consistent_label: consistent_count += 1
                    else: consistent_label=lbl; consistent_count=1

                    if consistent_count >= CONSISTENCY_REQUIRED and conf >= EARLY_TRIGGER_CONF:
                        candidates = generate_candidates(smooth_frames(sign_frames))
                        final_lbl, final_conf, method = select_from_candidates(candidates, word_buffer)
                        last_top3=[(c["label"],c["conf"]) for c in candidates]
                        last_seq_top3=last_top3; last_end_top3=[]
                        # Post-hoc gates（提早觸發路徑）
                        _fsrc = (np.array(list(stable_buffer)) if len(stable_buffer) >= 1
                                 else np.array(sign_frames[-8:]) if sign_frames
                                 else np.empty((0, 398), dtype=np.float32))
                        final_lbl, final_conf, method, _ = apply_gates_posthoc(
                            final_lbl, final_conf, method, candidates, _fsrc)
                        result_until=now+RESULT_SHOW_SEC; preview_top3=[]
                        if final_lbl != last_pred and final_lbl != "_unknown_":
                            word_buffer.append(final_lbl); last_pred=final_lbl
                            print(f"  [{final_conf*100:.0f}%] {final_lbl}  ({len(sign_frames)}幀, 提早觸發/{method})")
                            pending_log = log_prediction(final_lbl, final_conf, method, last_top3)
                        sign_frames=[]; stop_frames_cnt=0; stable_buffer.clear()
                        consistent_label=""; consistent_count=0; seg_state="IDLE"; triggered=True

                # ── 路徑 1.5：結尾姿勢高信心提早觸發（靜止第3幀即檢查）──
                endpose_early = False
                if (not triggered and
                    stop_frames_cnt == EARLY_ENDPOSE_MIN_STABLE and
                    len(sign_frames) >= MIN_FRAMES and
                    len(stable_buffer) >= EARLY_ENDPOSE_MIN_STABLE):
                    avg_feat_chk = np.concatenate([np.mean(np.array(stable_buffer), axis=0), np.zeros(6, dtype=np.float32)])
                    end_conf_chk = probs_to_top3(get_probs([avg_feat_chk] * SEQ_LEN))[0][1]
                    if end_conf_chk >= EARLY_ENDPOSE_CONF:
                        endpose_early = True
                        print(f"  [endpose提早] 靜止{stop_frames_cnt}幀 endpose={end_conf_chk*100:.0f}%")

                # ── 路徑二：停止觸發 → 多候選 + LLM 評分 ──
                if not triggered and (
                    len(sign_frames) >= SEQ_LEN or
                    endpose_early or
                    (stop_frames_cnt >= STOP_FRAMES and len(sign_frames) >= MIN_FRAMES)
                ):
                    use_endpose = (stop_frames_cnt >= STOP_FRAMES or endpose_early) and len(stable_buffer) >= 3
                    smoothed_frames = smooth_frames(sign_frames)

                    if use_endpose:
                        # 多候選皆使用 endpose 合併
                        def _predict_endpose_frames(frames):
                            lbl, conf, top3, seq_top3, end_top3 = predict_with_endpose(frames, stable_buffer)
                            return lbl, conf, top3
                        n = len(smoothed_frames)
                        candidates = []
                        for key, frames in [
                            ("A", smoothed_frames),
                            ("B", smoothed_frames[:n - CANDIDATE_OFFSET] if n - CANDIDATE_OFFSET >= MIN_FRAMES else None),
                            ("C", smoothed_frames[CANDIDATE_OFFSET:]       if n - CANDIDATE_OFFSET >= MIN_FRAMES else None),
                        ]:
                            if frames is None: continue
                            lbl, conf, top3 = _predict_endpose_frames(frames)
                            candidates.append({"key": key, "label": lbl, "conf": conf, "top3": top3})
                        _, _, _, last_seq_top3, last_end_top3 = predict_with_endpose(sign_frames, stable_buffer)
                        trigger_note = "停止+結尾合併"
                    else:
                        candidates = generate_candidates(smoothed_frames)
                        last_seq_top3 = candidates[0]["top3"]; last_end_top3 = []
                        trigger_note = "滿幀觸發"

                    final_lbl, final_conf, method = select_from_candidates(candidates, word_buffer)
                    last_top3=[(c["label"],c["conf"]) for c in candidates]
                    # 動作幅度（debug + 是偵測共用）
                    _xstd = _ystd = 0.0
                    if len(sign_frames) > 2:
                        _sf = np.array(sign_frames)
                        _xstd = float(np.std(_sf[:, 63])); _ystd = float(np.std(_sf[:, 64]))
                        print(f"    wrist_std x={_xstd:.3f} y={_ystd:.3f}  ({len(sign_frames)}幀)")
                    # Post-hoc gates（停止觸發路徑）
                    _fsrc = (np.array(list(stable_buffer)) if len(stable_buffer) >= 1
                             else np.array(sign_frames[-8:]) if sign_frames
                             else np.empty((0, 398), dtype=np.float32))
                    final_lbl, final_conf, method, _avg = apply_gates_posthoc(
                        final_lbl, final_conf, method, candidates, _fsrc)

                    # ── 是 專項偵測（最高優先）─────────────────────────────
                    # 來源一：候選清單（candidates top3）中「是」≥ 8%
                    # 來源二：直接查模型機率向量「是」≥ 5%
                    # 任一條件成立 → 強制輸出「是」
                    _si_cand_conf = 0.0
                    for _c in candidates:
                        for _lbl, _cf in _c.get("top3", []):
                            if _lbl == "是":
                                _si_cand_conf = max(_si_cand_conf, _cf)
                    if sign_frames:
                        _arr404 = add_cumulative(
                            np.array(sign_frames, dtype=np.float32))
                        _raw_p = get_probs(_arr404)
                        _raw_p = apply_feature_gates(
                            _raw_p, _gate_feat(_arr404), _arr404)
                        _si_idx = np.where(le.classes_ == "是")[0]
                        _si_raw_conf = float(_raw_p[_si_idx[0]]) if len(_si_idx) else 0.0
                    else:
                        _si_raw_conf = 0.0
                    print(f"    是偵測: 候選={_si_cand_conf*100:.1f}%  直查={_si_raw_conf*100:.1f}%  "
                          f"(當前={final_lbl} {final_conf*100:.1f}%)")
                    if _si_cand_conf >= 0.08 or _si_raw_conf >= 0.05:
                        _si_best = max(_si_cand_conf, _si_raw_conf)
                        final_lbl, final_conf = "是", _si_best
                        method += "+是偵"

                    # 個別詞彙信心門檻
                    _PER_SIGN_THRESH = {
                        "喜歡": 0.11,
                        "是":   0.08,
                        "找":   0.14,
                        "再見": 0.15,
                    }
                    # 是偵測 bypass：專項偵測啟動時幾乎不設門檻
                    if "+是偵" in method:
                        _thresh = 0.03
                    else:
                        _thresh = _PER_SIGN_THRESH.get(final_lbl, CONF_THRESHOLD)
                    result_until=now+RESULT_SHOW_SEC; preview_top3=[]
                    if final_conf >= _thresh and final_lbl != last_pred and final_lbl != "_unknown_":
                        word_buffer.append(final_lbl); last_pred=final_lbl
                        print(f"  [{final_conf*100:.0f}%] {final_lbl}  ({len(sign_frames)}幀, {trigger_note}/{method})")
                        pending_log = log_prediction(final_lbl, final_conf, method, last_top3)
                    else:
                        print(f"  [skip {final_conf*100:.0f}%] {final_lbl}  ({len(sign_frames)}幀)")
                    sign_frames=[]; stop_frames_cnt=0; stable_buffer.clear()
                    consistent_label=""; consistent_count=0; seg_state="IDLE"

    if not hand_detected and not paused:
        no_hand_count += 1; prev_wrist=None; prev_feat_delta=None; stop_frames_cnt=0
        if seg_state=="COLLECTING" and no_hand_count > NO_HAND_TOLERANCE:
            seg_state="IDLE"; sign_frames=[]; preview_top3=[]
            consistent_label=""; consistent_count=0; no_hand_count=0

    # ── UI ──
    if paused:           status_text="[ 暫停 ]";   status_color=(100,100,100)
    elif seg_state=="COLLECTING":
                         status_text=f"收集中 {len(sign_frames)}/{SEQ_LEN}"; status_color=(0,80,255)
                         cv2.circle(frame,(620,20),10,(0,0,255),-1)
    elif hand_detected:  status_text="偵測到手";    status_color=(0,200,120)
    else:                status_text="等待手勢";    status_color=(120,120,120)

    frame=put_chinese(frame,status_text,(10,8),size=22,color=status_color)
    # 詞彙逐行顯示（每行最多 8 個，保留所有詞彙）
    if word_buffer:
        _chunk = 8
        _lines = [word_buffer[i:i+_chunk] for i in range(0, len(word_buffer), _chunk)]
        for _li, _wline in enumerate(_lines):
            frame=put_chinese(frame,"  ".join(_wline),(10,38+_li*24),size=20,color=(255,255,100))
    else:
        frame=put_chinese(frame,"（空）",(10,38),size=20,color=(255,255,100))

    _word_lines = max(1, -(-len(word_buffer) // 8))  # 向上取整，至少 1 行
    _cand_y0 = 38 + _word_lines * 24 + 4
    if now < result_until:
        if last_top3:
            frame=put_chinese(frame,"合併:",(10,_cand_y0),size=17,color=(200,200,200))
            for i,(lbl,conf) in enumerate(last_top3[:3]):
                c=(200,255,200) if i==0 else (140,140,140)
                frame=put_chinese(frame,f"{i+1}.{lbl} {conf*100:.0f}%",(60,_cand_y0+i*24),size=18,color=c)
        if last_seq_top3:
            frame=put_chinese(frame,"序列:",(10,_cand_y0+80),size=17,color=(150,180,255))
            for i,(lbl,conf) in enumerate(last_seq_top3[:2]):
                c=(150,180,255) if i==0 else (100,100,150)
                frame=put_chinese(frame,f"{lbl} {conf*100:.0f}%",(60,_cand_y0+80+i*22),size=17,color=c)
        if last_end_top3:
            frame=put_chinese(frame,"結尾:",(10,_cand_y0+130),size=17,color=(255,180,100))
            for i,(lbl,conf) in enumerate(last_end_top3[:2]):
                c=(255,200,120) if i==0 else (150,120,80)
                frame=put_chinese(frame,f"{lbl} {conf*100:.0f}%",(60,_cand_y0+130+i*22),size=17,color=c)
    elif seg_state=="COLLECTING" and preview_top3:
        for i,(lbl,conf) in enumerate(preview_top3):
            c=(120,160,120) if i==0 else (90,90,90)
            frame=put_chinese(frame,f"{i+1}.{lbl} {conf*100:.0f}%",(10,_cand_y0+i*24),size=17,color=c)

    # 靜止進度條
    bar=min(stop_frames_cnt,STOP_FRAMES)
    cv2.rectangle(frame,(480,8),(480+int(bar/STOP_FRAMES*140),22),(0,200,150),-1)
    cv2.rectangle(frame,(480,8),(620,22),(80,80,80),1)
    frame=put_chinese(frame,f"靜止{stop_frames_cnt}/{STOP_FRAMES}",(480,24),size=15,color=(150,200,150))

    frame=put_chinese(frame,"Q:退出  C:清空  空白:暫停  Y/N:正確/錯誤",(10,455),size=17,color=(160,160,160))
    cv2.imshow("EndPose Recognizer (實驗)",frame)

    key=cv2.waitKey(1)&0xFF
    if key==ord('q'): break
    elif key==ord('c'):
        word_buffer.clear(); last_top3=[]; last_seq_top3=[]; last_end_top3=[]
        sign_frames=[]; stable_buffer.clear(); print("  已清空")
    elif key==ord(' '):
        paused=not paused; print(f"  {'暫停' if paused else '繼續'}")
    elif key==ord('y') and pending_log:
        row = {**pending_log, "verified": "correct"}
        _write_log(row)
        print(f"  ✓ 記錄正確：{pending_log['label']}  → 已存入 {LOG_PATH}")
        pending_log = None
    elif key==ord('n') and pending_log:
        row = {**pending_log, "verified": "wrong"}
        _write_log(row)
        print(f"  ✗ 記錄錯誤：{pending_log['label']}  → 已存入 {LOG_PATH}")
        pending_log = None

cap.release()
cv2.destroyAllWindows()
if word_buffer:
    print(f"\n最終詞彙：{' '.join(word_buffer)}")
