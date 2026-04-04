"""
app.py  —  台灣手語辨識 Web Server
Flask + SocketIO

瀏覽器每 100ms 送一幀 base64 JPEG → 伺服器跑 MediaPipe + BiLSTM → 推播結果
"""

import cv2
import numpy as np
import base64
import torch
import torch.nn as nn
import pickle
import os
from collections import deque
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import urllib.request
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import lm_selector

# ────────────────────────────────────────────────────────────
#  模型定義
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
        attn = torch.softmax(self.attention(lstm_out), dim=1)
        return self.classifier((attn * lstm_out).sum(dim=1))

# ────────────────────────────────────────────────────────────
#  設定
# ────────────────────────────────────────────────────────────
MODEL_PATH  = "models/lstm_best.pth"
CONFIG_PATH = "models/lstm_config.pkl"
LABEL_PATH  = "data/seq_label_encoder.pkl"
HAND_MODEL  = "hand_landmarker.task"
POSE_MODEL  = "pose_landmarker_lite.task"

W_HANDSHAPE=6.0; W_FINGERTIP=4.0; W_FINGER=3.0; W_PALM_DIR=3.0; W_BINARY=10.0
W_POSITION=4.0;  W_ARM=2.5;       W_FACE=5.0;   W_ORIENT=2.0; W_ANCHOR=1.5; W_POINTING=8.0
FINGERTIPS={4,8,12,16,20}
TIP_IDX=[4,8,12,16,20]; MCP_IDX=[1,5,9,13,17]
FINGER_JOINTS=[(2,3,4),(5,6,7),(9,10,11),(13,14,15),(17,18,19)]
EXTEND_ANGLE=140.0

NOSE=0; MOUTH_L=9; MOUTH_R=10; L_EAR=7; R_EAR=8
L_SHOULDER=11; R_SHOULDER=12; L_ELBOW=13; R_ELBOW=14
L_WRIST_POSE=15; R_WRIST_POSE=16; L_HIP=23; R_HIP=24

CONF_THRESHOLD=0.40; EARLY_TRIGGER_CONF=0.75; MIN_FRAMES=12
PREVIEW_INTERVAL=4; CONSISTENCY_REQUIRED=3
STOP_THRESHOLD=0.01; STOP_FRAMES=5; NO_HAND_TOLERANCE=5
SEQ_W=0.5; END_W=0.5; STABLE_BUFFER=8
LM_CONF_THRESHOLD=0.70; CANDIDATE_OFFSET=4; RAMP_FRAMES=10
EARLY_ENDPOSE_CONF=0.82; EARLY_ENDPOSE_MIN_STABLE=3

LABEL_ALIASES = {"現在":"現在／今天","安靜":"安靜／平安","相見":"相見／見面"}

# ────────────────────────────────────────────────────────────
#  特徵工程
# ────────────────────────────────────────────────────────────
def compute_finger_extensions(lm63):
    lms=lm63.reshape(21,3); wrist=lms[0]
    scores=np.empty(5,dtype=np.float32)
    for fi,(ti,mi) in enumerate(zip(TIP_IDX,MCP_IDX)):
        scores[fi]=np.linalg.norm(lms[ti]-wrist)/(np.linalg.norm(lms[mi]-wrist)+1e-6)
    return scores

def compute_pointing_dir(lm63):
    """食指指向方向（3D 正規化向量）：normalize(tip[8] - MCP[5])"""
    lms=lm63.reshape(21,3)
    d=lms[8]-lms[5]; return (d/(np.linalg.norm(d)+1e-6)).astype(np.float32)

def compute_finger_binary(lm63):
    """5-dim: 1=伸出, 0=彎曲（純 PIP 關節角度）"""
    lms=lm63.reshape(21,3)
    binary=np.empty(5,dtype=np.float32)
    for fi,(a,b,c) in enumerate(FINGER_JOINTS):
        v1=lms[a]-lms[b]; v2=lms[c]-lms[b]
        cos_a=np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)+1e-6)
        binary[fi]=1.0 if np.degrees(np.arccos(np.clip(cos_a,-1.0,1.0)))>EXTEND_ANGLE else 0.0
    return binary

def compute_pinch_dist(lm63):
    """捏合距離：||拇指尖-食指尖|| / 掌長"""
    lms=lm63.reshape(21,3)
    dist=np.linalg.norm(lms[4]-lms[8])/(np.linalg.norm(lms[9]-lms[0])+1e-6)
    return np.array([min(dist,3.0)],dtype=np.float32)

def apply_feature_weights(feat):
    out=feat.copy()
    for offset in (0,68):
        wx,wy=feat[offset],feat[offset+1]; mx,my=feat[offset+27],feat[offset+28]
        scale=np.sqrt((mx-wx)**2+(my-wy)**2)+1e-6
        for i in range(21):
            b=offset+i*3; w=W_FINGERTIP if i in FINGERTIPS else W_FINGER
            out[b]=(feat[b]-wx)/scale*w; out[b+1]=(feat[b+1]-wy)/scale*w; out[b+2]=feat[b+2]*w
        out[offset+66]=feat[offset+66]*W_PALM_DIR; out[offset+67]=feat[offset+67]*W_PALM_DIR
        out[offset+63]=feat[offset+63]*W_POSITION; out[offset+64]=feat[offset+64]*W_POSITION
        out[offset+65]=feat[offset+65]*W_POSITION
    out[136:146]=feat[136:146]*W_ANCHOR; out[149:161]=feat[149:161]*W_ARM
    out[161:167]=feat[161:167]*W_FACE;   out[167:169]=feat[167:169]*W_ORIENT
    out[169:179]=feat[169:179]*W_HANDSHAPE
    out[179:189]=feat[179:189]*W_BINARY       # 手指伸展二值
    out[189:195]=feat[189:195]*W_POINTING    # 食指指向方向
    out[195:197]=feat[195:197]*W_BINARY      # 捏合距離
    out[197:199]=feat[197:199]*W_BINARY      # 手部偵測旗標
    return out

# ────────────────────────────────────────────────────────────
#  下載模型
# ────────────────────────────────────────────────────────────
def _dl(path, url):
    if not os.path.exists(path):
        print(f"Downloading {path}..."); urllib.request.urlretrieve(url, path); print(f"[OK] {path}")

_dl(HAND_MODEL,"https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
_dl(POSE_MODEL,"https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task")

# ────────────────────────────────────────────────────────────
#  初始化 MediaPipe
# ────────────────────────────────────────────────────────────
hand_detector = vision.HandLandmarker.create_from_options(
    vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=HAND_MODEL),
        num_hands=2, min_hand_detection_confidence=0.4,
        min_hand_presence_confidence=0.4, min_tracking_confidence=0.4,
    )
)
pose_detector = vision.PoseLandmarker.create_from_options(
    vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=POSE_MODEL),
        min_pose_detection_confidence=0.4, min_pose_presence_confidence=0.4,
        min_tracking_confidence=0.4,
    )
)
print("[OK] MediaPipe ready")

# ────────────────────────────────────────────────────────────
#  載入 BiLSTM
# ────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

with open(CONFIG_PATH,"rb") as f: config=pickle.load(f)
with open(LABEL_PATH, "rb") as f: le    =pickle.load(f)

bilstm = SignLanguageBiLSTM(
    input_dim=config["input_dim"], num_classes=config["num_classes"],
    hidden_dim=config["hidden_dim"], num_layers=config["num_layers"], dropout=config["dropout"],
).to(device)
ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
bilstm.load_state_dict(ckpt["model_state_dict"])
bilstm.eval()

feat_mean = torch.FloatTensor(config["feat_mean"]).to(device)
feat_std  = torch.FloatTensor(config["feat_std"]).to(device)
SEQ_LEN   = config["seq_len"]
print(f"[OK] BiLSTM loaded (Val Acc: {ckpt['val_acc']*100:.2f}%)")

# ────────────────────────────────────────────────────────────
#  推論
# ────────────────────────────────────────────────────────────
@torch.no_grad()
def get_probs(frames_list):
    seq = np.array(frames_list, dtype=np.float32)
    T = len(seq)
    if T != SEQ_LEN:
        seq = seq[np.linspace(0, T-1, SEQ_LEN, dtype=int)]
    # 若尚未加 cumulative（398-dim），自動補上
    if seq.shape[1] == 398:
        right_start = seq[0:1, 63:66]
        left_start  = seq[0:1, 131:134]
        cum_r = seq[:, 63:66]   - right_start
        cum_l = seq[:, 131:134] - left_start
        seq = np.concatenate([seq, cum_r, cum_l], axis=1)
    x = (torch.FloatTensor(seq).unsqueeze(0).to(device) - feat_mean) / feat_std
    return torch.softmax(bilstm(x), dim=1)[0]

def probs_to_top3(probs):
    top3_conf, top3_idx = probs.topk(3)
    return [(LABEL_ALIASES.get(le.classes_[i.item()], le.classes_[i.item()]), round(c.item()*100))
            for i, c in zip(top3_idx, top3_conf)]

def predict_frames_app(frames):
    """回傳 (label, conf_float, top3_percent) — conf_float 用於 LM 評分比較"""
    probs = get_probs(frames)
    top3  = probs_to_top3(probs)
    return top3[0][0], top3[0][1] / 100.0, top3

def add_cumulative_app(frames_arr):
    """frames_arr: (T, D) → (T, D+6)，追加右/左手腕累積位移"""
    right_start = frames_arr[0:1, 63:66]
    left_start  = frames_arr[0:1, 131:134]
    cum_r = frames_arr[:, 63:66]   - right_start
    cum_l = frames_arr[:, 131:134] - left_start
    return np.concatenate([frames_arr, cum_r, cum_l], axis=1)

def generate_candidates_app(frames):
    """產生 A/B/C 候選切法，所有候選先丟棄前 RAMP_FRAMES 過渡幀"""
    candidates = []
    frames = frames[RAMP_FRAMES:] if len(frames) > RAMP_FRAMES + MIN_FRAMES else frames
    frames = add_cumulative_app(np.array(frames))
    n = len(frames)
    for key, slc in [
        ("A", frames),
        ("B", frames[:n - CANDIDATE_OFFSET] if n - CANDIDATE_OFFSET >= MIN_FRAMES else None),
        ("C", frames[CANDIDATE_OFFSET:]      if n - CANDIDATE_OFFSET >= MIN_FRAMES else None),
    ]:
        if slc is None: continue
        lbl, conf, top3 = predict_frames_app(slc)
        candidates.append({"key": key, "label": lbl, "conf": conf, "top3": top3})
    return candidates

def select_from_candidates_app(candidates, word_buffer):
    """回傳 (final_label, conf_percent, method_str)"""
    best = max(candidates, key=lambda c: c["conf"])
    if best["conf"] >= LM_CONF_THRESHOLD:
        return best["label"], round(best["conf"]*100), f"候選{best['key']}"
    lbl = lm_selector.lm_select(candidates, word_buffer)
    conf = next((c["conf"] for c in candidates if c["label"] == lbl), best["conf"])
    return lbl, round(conf*100), "LM評分"

# ────────────────────────────────────────────────────────────
#  特徵提取（單幀，從 BGR numpy array）
# ────────────────────────────────────────────────────────────
def extract_feat(frame_bgr):
    rgb    = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    hand_result = hand_detector.detect(mp_img)
    pose_result = pose_detector.detect(mp_img)

    if not hand_result.hand_landmarks:
        return None, False

    right_hand = left_hand = None
    for i, cls in enumerate(hand_result.handedness):
        if cls[0].category_name == "Left": right_hand = hand_result.hand_landmarks[i]
        else:                               left_hand  = hand_result.hand_landmarks[i]

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
        r_wrist_p=pose[R_WRIST_POSE]; l_wrist_p=pose[L_WRIST_POSE]
        r_sho_lm=pose[R_SHOULDER]; l_sho_lm=pose[L_SHOULDER]
        mouth_c=np.array([(pose[MOUTH_L].x+pose[MOUTH_R].x)/2,
                           (pose[MOUTH_L].y+pose[MOUTH_R].y)/2],dtype=np.float32)
        r_ear_pt=np.array([pose[R_EAR].x,pose[R_EAR].y],dtype=np.float32)
        l_ear_pt=np.array([pose[L_EAR].x,pose[L_EAR].y],dtype=np.float32)
    else:
        nose=np.array([0.5,0.3],dtype=np.float32); l_sho=np.array([0.4,0.5],dtype=np.float32)
        r_sho=np.array([0.6,0.5],dtype=np.float32); mid_sho=np.array([0.5,0.5],dtype=np.float32)
        sho_width=0.3; body_anchors=np.zeros(10,dtype=np.float32)
        r_elbow_lm=l_elbow_lm=r_wrist_p=l_wrist_p=None
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

    def arm_feat(elbow,wrist_lm,sho_lm):
        if elbow is None or sho_lm is None: return np.zeros(6,dtype=np.float32)
        e=np.array([elbow.x,elbow.y],dtype=np.float32)
        s=np.array([sho_lm.x,sho_lm.y],dtype=np.float32)
        er=np.array([(e[0]-mid_sho[0])/sho_width,(e[1]-mid_sho[1])/sho_width],dtype=np.float32)
        u=e-s; u=u/(np.linalg.norm(u)+1e-6)
        if wrist_lm:
            wv=np.array([wrist_lm.x,wrist_lm.y],dtype=np.float32)
            f=wv-e; f=f/(np.linalg.norm(f)+1e-6)
        else: f=np.zeros(2,dtype=np.float32)
        return np.concatenate([er,u,f])

    def face_dist(hand,ear_pt):
        if hand is None: return np.zeros(3,dtype=np.float32)
        w=np.array([hand[0].x,hand[0].y],dtype=np.float32)
        return np.array([np.linalg.norm(w-nose)/sho_width,
                         np.linalg.norm(w-mouth_c)/sho_width,
                         np.linalg.norm(w-ear_pt)/sho_width],dtype=np.float32)

    sho_vec=r_sho-l_sho; theta=np.arctan2(sho_vec[1],sho_vec[0])
    body_orient=np.array([np.cos(theta),np.sin(theta)],dtype=np.float32)

    raw167=np.concatenate([
        hand_feat(right_hand),hand_feat(left_hand),
        body_anchors,np.zeros(3,dtype=np.float32),
        arm_feat(r_elbow_lm,r_wrist_p,r_sho_lm),
        arm_feat(l_elbow_lm,l_wrist_p,l_sho_lm),
        face_dist(right_hand,r_ear_pt),face_dist(left_hand,l_ear_pt),
    ])
    ext_r=compute_finger_extensions(raw167[:63])    if right_hand else np.zeros(5,dtype=np.float32)
    ext_l=compute_finger_extensions(raw167[68:131]) if left_hand  else np.zeros(5,dtype=np.float32)
    bin_r  =compute_finger_binary(raw167[:63])        if right_hand else np.zeros(5,dtype=np.float32)
    bin_l  =compute_finger_binary(raw167[68:131])     if left_hand  else np.zeros(5,dtype=np.float32)
    ptr_r  =compute_pointing_dir(raw167[:63])          if right_hand else np.zeros(3,dtype=np.float32)
    ptr_l  =compute_pointing_dir(raw167[68:131])       if left_hand  else np.zeros(3,dtype=np.float32)
    pinch_r=compute_pinch_dist(raw167[:63])            if right_hand else np.array([3.0],dtype=np.float32)
    pinch_l=compute_pinch_dist(raw167[68:131])         if left_hand  else np.array([3.0],dtype=np.float32)
    is_right=np.array([1.0 if right_hand else 0.0],dtype=np.float32)
    is_left =np.array([1.0 if left_hand  else 0.0],dtype=np.float32)
    feat=np.concatenate([raw167,body_orient,ext_r,ext_l,bin_r,bin_l,ptr_r,ptr_l,pinch_r,pinch_l,is_right,is_left])
    return apply_feature_weights(feat), True

# ────────────────────────────────────────────────────────────
#  辨識狀態（單一使用者）
# ────────────────────────────────────────────────────────────
state = {
    "seg_state":        "IDLE",
    "sign_frames":      [],
    "stable_buffer":    deque(maxlen=STABLE_BUFFER),
    "no_hand_count":    0,
    "stop_frames_cnt":  0,
    "prev_wrist":       None,
    "prev_feat_delta":  None,   # ← 前一幀特徵，用於計算 delta
    "preview_top3":     [],
    "consistent_label": "",
    "consistent_count": 0,
    "word_buffer":      [],
    "last_pred":        None,
}

def reset_seg():
    state["seg_state"]        = "IDLE"
    state["sign_frames"]      = []
    state["stable_buffer"].clear()
    state["stop_frames_cnt"]  = 0
    state["preview_top3"]     = []
    state["consistent_label"] = ""
    state["consistent_count"] = 0

# ────────────────────────────────────────────────────────────
#  Flask + SocketIO
# ────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["SECRET_KEY"] = "tsl-demo"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

@app.route("/")
def index():
    return render_template("index.html")

@socketio.on("frame")
def handle_frame(data):
    """
    接收 base64 JPEG → 提取特徵 → 狀態機 → 若有結果推播 result 事件
    """
    # 解碼圖片
    img_bytes = base64.b64decode(data["image"].split(",")[-1])
    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return

    feat, hand_detected = extract_feat(frame)

    if hand_detected:
        # Delta（速度）特徵：197 → 394 維
        prev_fd = state["prev_feat_delta"]
        delta = feat - prev_fd if prev_fd is not None else np.zeros_like(feat)
        state["prev_feat_delta"] = feat.copy()
        feat = np.concatenate([feat, delta])

        state["no_hand_count"] = 0

        # 手腕速度（仍用原始 rel_wrist，索引不變）
        cur_wrist = feat[63:65].copy()  # right hand rel_wrist x,y
        if state["prev_wrist"] is not None:
            speed = float(np.linalg.norm(cur_wrist - state["prev_wrist"]))
        else:
            speed = 1.0
        state["prev_wrist"] = cur_wrist

        if speed < STOP_THRESHOLD:
            state["stop_frames_cnt"] += 1
            state["stable_buffer"].append(feat.copy())
        else:
            state["stop_frames_cnt"] = 0

        # 狀態機
        if state["seg_state"] == "IDLE":
            state["seg_state"]    = "COLLECTING"
            state["sign_frames"]  = [feat]

        elif state["seg_state"] == "COLLECTING":
            state["sign_frames"].append(feat)
            triggered = False

            # 預覽
            if len(state["sign_frames"]) >= MIN_FRAMES and len(state["sign_frames"]) % PREVIEW_INTERVAL == 0:
                probs = get_probs(state["sign_frames"])
                top3  = probs_to_top3(probs)
                state["preview_top3"] = top3
                lbl = top3[0][0]
                if lbl == state["consistent_label"]:
                    state["consistent_count"] += 1
                else:
                    state["consistent_label"] = lbl
                    state["consistent_count"]  = 1

                if state["consistent_count"] >= CONSISTENCY_REQUIRED and top3[0][1] >= EARLY_TRIGGER_CONF*100:
                    candidates = generate_candidates_app(state["sign_frames"])
                    final_lbl, final_conf, method = select_from_candidates_app(candidates, state["word_buffer"])
                    emit("result", {
                        "word": final_lbl, "conf": final_conf,
                        "top3": [(c["label"], round(c["conf"]*100)) for c in candidates],
                        "trigger": f"early/{method}",
                        "words": state["word_buffer"],
                    })
                    if final_lbl != state["last_pred"] and final_lbl != "_unknown_":
                        state["word_buffer"].append(final_lbl)
                        state["last_pred"] = final_lbl
                        emit("words", {"words": state["word_buffer"]})
                    reset_seg(); triggered = True

            # 結尾姿勢高信心提早觸發（靜止第3幀即檢查）
            endpose_early = False
            if (not triggered and
                state["stop_frames_cnt"] == EARLY_ENDPOSE_MIN_STABLE and
                len(state["sign_frames"]) >= MIN_FRAMES and
                len(state["stable_buffer"]) >= EARLY_ENDPOSE_MIN_STABLE):
                avg_chk = np.concatenate([np.mean(np.array(state["stable_buffer"]), axis=0), np.zeros(6, dtype=np.float32)])
                end_conf_chk = probs_to_top3(get_probs([avg_chk] * SEQ_LEN))[0][1]
                if end_conf_chk >= EARLY_ENDPOSE_CONF * 100:
                    endpose_early = True

            # 停止或滿幀觸發 → 多候選 + LLM 評分
            if not triggered and (
                len(state["sign_frames"]) >= SEQ_LEN or
                endpose_early or
                (state["stop_frames_cnt"] >= STOP_FRAMES and len(state["sign_frames"]) >= MIN_FRAMES)
            ):
                use_endpose = (state["stop_frames_cnt"] >= STOP_FRAMES or endpose_early) and len(state["stable_buffer"]) >= 3
                frames = state["sign_frames"]
                n = len(frames)

                if use_endpose:
                    avg_feat = np.concatenate([np.mean(np.array(state["stable_buffer"]), axis=0), np.zeros(6, dtype=np.float32)])
                    def _endpose_predict(slc):
                        sp = get_probs(slc)
                        ep = get_probs([avg_feat] * SEQ_LEN)
                        combined = sp * SEQ_W + ep * END_W
                        buf_arr = np.array(state["stable_buffer"])
                        # Gate 1: 雙手 gate — 左手幀比例<40% 排除穩定雙手詞彙
                        left_rate = (buf_arr[:, 198] > W_BINARY * 0.5).mean()
                        if left_rate < 0.4:
                            BILATERAL_ONLY = {"對不起", "認真", "人", "聽人", "晚上"}
                            p = combined.cpu().numpy().copy()
                            for word in BILATERAL_ONLY:
                                idx_arr = np.where(le.classes_ == word)[0]
                                if len(idx_arr): p[idx_arr[0]] *= 0.02
                            p /= (p.sum() + 1e-9)
                            combined = torch.FloatTensor(p).to(combined.device)
                        # Gate 2: 說 z方向 gate — 食指未明顯朝向鏡頭時壓制「說」
                        point_z_w = float(buf_arr.mean(axis=0)[191])
                        if point_z_w > -0.18 * W_POINTING:
                            p = combined.cpu().numpy().copy()
                            idx_arr = np.where(le.classes_ == "說")[0]
                            if len(idx_arr): p[idx_arr[0]] *= 0.1
                            p /= (p.sum() + 1e-9)
                            combined = torch.FloatTensor(p).to(combined.device)
                        top3 = probs_to_top3(combined)
                        return top3[0][0], top3[0][1] / 100.0, top3
                    candidates = []
                    for key, slc in [
                        ("A", frames),
                        ("B", frames[:n - CANDIDATE_OFFSET] if n - CANDIDATE_OFFSET >= MIN_FRAMES else None),
                        ("C", frames[CANDIDATE_OFFSET:]      if n - CANDIDATE_OFFSET >= MIN_FRAMES else None),
                    ]:
                        if slc is None: continue
                        lbl, conf, top3 = _endpose_predict(slc)
                        candidates.append({"key": key, "label": lbl, "conf": conf, "top3": top3})
                    seq_top3 = probs_to_top3(get_probs(frames))
                    end_top3 = probs_to_top3(get_probs([avg_feat] * SEQ_LEN))
                    trigger  = "stop+endpose"
                else:
                    candidates = generate_candidates_app(frames)
                    seq_top3   = candidates[0]["top3"]; end_top3 = []
                    trigger    = "full"

                final_lbl, final_conf, method = select_from_candidates_app(candidates, state["word_buffer"])
                if final_conf >= CONF_THRESHOLD*100 and final_lbl != state["last_pred"] and final_lbl != "_unknown_":
                    state["word_buffer"].append(final_lbl)
                    state["last_pred"] = final_lbl
                    emit("words", {"words": state["word_buffer"]})

                emit("result", {
                    "word": final_lbl, "conf": final_conf,
                    "top3": [(c["label"], round(c["conf"]*100)) for c in candidates],
                    "seq_top3": seq_top3, "end_top3": end_top3,
                    "trigger": f"{trigger}/{method}", "words": state["word_buffer"],
                })
                reset_seg()

        # 推播即時預覽
        emit("preview", {
            "top3": state["preview_top3"],
            "collecting": state["seg_state"] == "COLLECTING",
            "frames":     len(state["sign_frames"]),
            "seq_len":    SEQ_LEN,
            "stop_cnt":   state["stop_frames_cnt"],
            "stop_max":   STOP_FRAMES,
        })

    else:
        state["no_hand_count"]   += 1
        state["prev_wrist"]       = None
        state["prev_feat_delta"]  = None   # 手消失時重置 delta
        state["stop_frames_cnt"]  = 0
        if state["seg_state"] == "COLLECTING" and state["no_hand_count"] > NO_HAND_TOLERANCE:
            reset_seg()

@socketio.on("clear")
def handle_clear():
    state["word_buffer"].clear()
    state["last_pred"] = None
    reset_seg()
    emit("words", {"words": []})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\nStarting server on port {port}...")
    socketio.run(app, host="0.0.0.0", port=port, debug=False, allow_unsafe_werkzeug=True)
