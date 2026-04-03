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

W_HANDSHAPE=6.0; W_FINGERTIP=4.0; W_FINGER=3.0; W_PALM_DIR=3.0
W_POSITION=2.5;  W_ARM=2.5;       W_FACE=3.0;   W_ORIENT=2.0; W_ANCHOR=1.5
FINGERTIPS={4,8,12,16,20}
TIP_IDX=[4,8,12,16,20]; MCP_IDX=[1,5,9,13,17]

NOSE=0; MOUTH_L=9; MOUTH_R=10; L_EAR=7; R_EAR=8
L_SHOULDER=11; R_SHOULDER=12; L_ELBOW=13; R_ELBOW=14
L_WRIST_POSE=15; R_WRIST_POSE=16; L_HIP=23; R_HIP=24

CONF_THRESHOLD=0.5; EARLY_TRIGGER_CONF=0.75; MIN_FRAMES=12
PREVIEW_INTERVAL=4; CONSISTENCY_REQUIRED=2
STOP_THRESHOLD=0.01; STOP_FRAMES=5; NO_HAND_TOLERANCE=5
SEQ_W=0.6; END_W=0.4; STABLE_BUFFER=8

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
    x = (torch.FloatTensor(seq).unsqueeze(0).to(device) - feat_mean) / feat_std
    return torch.softmax(bilstm(x), dim=1)[0]

def probs_to_top3(probs):
    top3_conf, top3_idx = probs.topk(3)
    return [(LABEL_ALIASES.get(le.classes_[i.item()], le.classes_[i.item()]), round(c.item()*100))
            for i, c in zip(top3_idx, top3_conf)]

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
    feat=np.concatenate([raw167,body_orient,ext_r,ext_l])
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
        state["no_hand_count"] = 0

        # 手腕速度
        # 用 feat 中的 rel_wrist（dims 63-64 for right hand）
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
                    emit("result", {
                        "word": lbl, "conf": top3[0][1],
                        "top3": top3, "trigger": "early",
                        "words": state["word_buffer"],
                    })
                    if lbl != state["last_pred"] and lbl != "_unknown_":
                        state["word_buffer"].append(lbl)
                        state["last_pred"] = lbl
                        emit("words", {"words": state["word_buffer"]})
                    reset_seg(); triggered = True

            # 停止或滿幀觸發
            if not triggered and (
                len(state["sign_frames"]) >= SEQ_LEN or
                (state["stop_frames_cnt"] >= STOP_FRAMES and len(state["sign_frames"]) >= MIN_FRAMES)
            ):
                use_endpose = state["stop_frames_cnt"] >= STOP_FRAMES and len(state["stable_buffer"]) >= 3

                if use_endpose:
                    seq_probs = get_probs(state["sign_frames"])
                    avg_feat  = np.mean(np.array(state["stable_buffer"]), axis=0)
                    end_probs = get_probs([avg_feat] * SEQ_LEN)
                    combined  = seq_probs * SEQ_W + end_probs * END_W
                    top3      = probs_to_top3(combined)
                    seq_top3  = probs_to_top3(seq_probs)
                    end_top3  = probs_to_top3(end_probs)
                    trigger   = "stop+endpose"
                else:
                    probs    = get_probs(state["sign_frames"])
                    top3     = probs_to_top3(probs)
                    seq_top3 = top3; end_top3 = []
                    trigger  = "full"

                lbl, conf = top3[0]
                if conf >= CONF_THRESHOLD*100 and lbl != state["last_pred"] and lbl != "_unknown_":
                    state["word_buffer"].append(lbl)
                    state["last_pred"] = lbl
                    emit("words", {"words": state["word_buffer"]})

                emit("result", {
                    "word": lbl, "conf": conf,
                    "top3": top3, "seq_top3": seq_top3, "end_top3": end_top3,
                    "trigger": trigger, "words": state["word_buffer"],
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
        state["no_hand_count"] += 1
        state["prev_wrist"]    = None
        state["stop_frames_cnt"] = 0
        if state["seg_state"] == "COLLECTING" and state["no_hand_count"] > NO_HAND_TOLERANCE:
            reset_seg()

@socketio.on("clear")
def handle_clear():
    state["word_buffer"].clear()
    state["last_pred"] = None
    reset_seg()
    emit("words", {"words": []})

if __name__ == "__main__":
    print("\nStarting server...")
    print("   http://localhost:5000")
    socketio.run(app, host="0.0.0.0", port=5000, debug=False)
