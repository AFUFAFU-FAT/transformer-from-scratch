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
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import torch
import torch.nn as nn
import pickle, time, os
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

FEAT_DIM    = 179

W_HANDSHAPE = 6.0; W_FINGERTIP = 4.0; W_FINGER = 3.0
W_PALM_DIR  = 3.0; W_POSITION  = 2.5; W_ARM    = 2.5
W_FACE = 3.0; W_ORIENT = 2.0; W_ANCHOR = 1.5
FINGERTIPS = {4, 8, 12, 16, 20}
TIP_IDX = [4, 8, 12, 16, 20]; MCP_IDX = [1, 5, 9, 13, 17]

NOSE = 0; MOUTH_L = 9; MOUTH_R = 10; L_EAR = 7; R_EAR = 8
L_SHOULDER=11; R_SHOULDER=12; L_ELBOW=13; R_ELBOW=14
L_WRIST_POSE=15; R_WRIST_POSE=16; L_HIP=23; R_HIP=24

CONF_THRESHOLD       = 0.5
EARLY_TRIGGER_CONF   = 0.75
MIN_FRAMES           = 12
PREVIEW_INTERVAL     = 4
CONSISTENCY_REQUIRED = 2
SENTENCE_PAUSE       = 4.0
NO_HAND_TOLERANCE    = 5
STOP_THRESHOLD       = 0.01
STOP_FRAMES          = 5
RESULT_SHOW_SEC      = 2.5

# ── 結尾合併設定 ──
SEQ_W          = 0.6    # 序列預測權重
END_W          = 0.4    # 結尾姿勢預測權重
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
    x = (torch.FloatTensor(seq).unsqueeze(0).to(device) - feat_mean) / feat_std
    return torch.softmax(model(x), dim=1)[0]

def probs_to_top3(probs):
    top3_conf, top3_idx = probs.topk(3)
    return [(LABEL_ALIASES.get(le.classes_[i.item()], le.classes_[i.item()]), c.item())
            for i, c in zip(top3_idx, top3_conf)]

def predict_frames(frames_list):
    probs = get_probs(frames_list)
    top3  = probs_to_top3(probs)
    return top3[0][0], top3[0][1], top3

@torch.no_grad()
def predict_with_endpose(sign_frames, stable_buffer):
    """
    合併：sequence_probs * SEQ_W + endpose_probs * END_W
    stable_buffer：deque of (FEAT_DIM,) 最近幾幀靜止特徵
    """
    seq_probs = get_probs(sign_frames)

    # 結尾姿勢：取 stable_buffer 的平均幀，重複 SEQ_LEN 次
    avg_feat  = np.mean(np.array(stable_buffer), axis=0)
    end_probs = get_probs([avg_feat] * SEQ_LEN)

    combined  = seq_probs * SEQ_W + end_probs * END_W

    top3_combined = probs_to_top3(combined)
    top3_seq      = probs_to_top3(seq_probs)
    top3_end      = probs_to_top3(end_probs)

    lbl, conf = top3_combined[0]
    return lbl, conf, top3_combined, top3_seq, top3_end

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
print("辨識中（Q=退出  C=清空  空白=暫停）")
print("-" * 50)

seg_state        = "IDLE"
sign_frames      = []
stable_buffer    = deque(maxlen=STABLE_BUFFER)   # ← 結尾 ring buffer
no_hand_count    = 0
stop_frames_cnt  = 0
prev_wrist       = None
preview_top3     = []
consistent_label = ""
consistent_count = 0
word_buffer      = []
last_top3        = []
last_seq_top3    = []   # ← 顯示用：序列預測
last_end_top3    = []   # ← 顯示用：結尾預測
result_until     = 0.0
last_pred        = None
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
        feat=np.concatenate([raw167,body_orient,ext_r,ext_l])
        feat=apply_feature_weights(feat)

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
                        last_top3=top3; last_seq_top3=top3; last_end_top3=[]
                        result_until=now+RESULT_SHOW_SEC; preview_top3=[]
                        if lbl != last_pred and lbl != "_unknown_":
                            word_buffer.append(lbl); last_pred=lbl
                            print(f"  [{conf*100:.0f}%] {lbl}  ({len(sign_frames)}幀, 提早觸發)")
                        sign_frames=[]; stop_frames_cnt=0; stable_buffer.clear()
                        consistent_label=""; consistent_count=0; seg_state="IDLE"; triggered=True

                # ── 路徑二：停止觸發 → 加入結尾姿勢合併 ──
                if not triggered and (
                    len(sign_frames) >= SEQ_LEN or
                    (stop_frames_cnt >= STOP_FRAMES and len(sign_frames) >= MIN_FRAMES)
                ):
                    use_endpose = stop_frames_cnt >= STOP_FRAMES and len(stable_buffer) >= 3

                    if use_endpose:
                        lbl, conf, top3, seq_top3, end_top3 = predict_with_endpose(sign_frames, stable_buffer)
                        last_seq_top3 = seq_top3
                        last_end_top3 = end_top3
                        trigger_note  = f"停止+結尾合併 seq={seq_top3[0][0]}({seq_top3[0][1]*100:.0f}%) end={end_top3[0][0]}({end_top3[0][1]*100:.0f}%)"
                    else:
                        lbl, conf, top3 = predict_frames(sign_frames)
                        last_seq_top3 = top3; last_end_top3 = []
                        trigger_note  = "滿幀觸發"

                    last_top3=top3; result_until=now+RESULT_SHOW_SEC; preview_top3=[]
                    if conf >= CONF_THRESHOLD and lbl != last_pred and lbl != "_unknown_":
                        word_buffer.append(lbl); last_pred=lbl
                        print(f"  [{conf*100:.0f}%] {lbl}  ({len(sign_frames)}幀, {trigger_note})")
                    else:
                        print(f"  [skip {conf*100:.0f}%] {lbl}  ({len(sign_frames)}幀)")
                    sign_frames=[]; stop_frames_cnt=0; stable_buffer.clear()
                    consistent_label=""; consistent_count=0; seg_state="IDLE"

    if not hand_detected and not paused:
        no_hand_count += 1; prev_wrist=None; stop_frames_cnt=0
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
    words_str="  ".join(word_buffer[-5:]) if word_buffer else "（空）"
    frame=put_chinese(frame,f"詞彙：{words_str}",(10,38),size=20,color=(255,255,100))

    if now < result_until:
        if last_top3:
            frame=put_chinese(frame,"合併:",(10,68),size=17,color=(200,200,200))
            for i,(lbl,conf) in enumerate(last_top3[:3]):
                c=(200,255,200) if i==0 else (140,140,140)
                frame=put_chinese(frame,f"{i+1}.{lbl} {conf*100:.0f}%",(60,68+i*24),size=18,color=c)
        if last_seq_top3:
            frame=put_chinese(frame,"序列:",(10,148),size=17,color=(150,180,255))
            for i,(lbl,conf) in enumerate(last_seq_top3[:2]):
                c=(150,180,255) if i==0 else (100,100,150)
                frame=put_chinese(frame,f"{lbl} {conf*100:.0f}%",(60,148+i*22),size=17,color=c)
        if last_end_top3:
            frame=put_chinese(frame,"結尾:",(10,198),size=17,color=(255,180,100))
            for i,(lbl,conf) in enumerate(last_end_top3[:2]):
                c=(255,200,120) if i==0 else (150,120,80)
                frame=put_chinese(frame,f"{lbl} {conf*100:.0f}%",(60,198+i*22),size=17,color=c)
    elif seg_state=="COLLECTING" and preview_top3:
        for i,(lbl,conf) in enumerate(preview_top3):
            c=(120,160,120) if i==0 else (90,90,90)
            frame=put_chinese(frame,f"{i+1}.{lbl} {conf*100:.0f}%",(10,68+i*24),size=17,color=c)

    # 靜止進度條
    bar=min(stop_frames_cnt,STOP_FRAMES)
    cv2.rectangle(frame,(480,8),(480+int(bar/STOP_FRAMES*140),22),(0,200,150),-1)
    cv2.rectangle(frame,(480,8),(620,22),(80,80,80),1)
    frame=put_chinese(frame,f"靜止{stop_frames_cnt}/{STOP_FRAMES}",(480,24),size=15,color=(150,200,150))

    frame=put_chinese(frame,"Q:退出  C:清空  空白:暫停",(10,455),size=17,color=(160,160,160))
    cv2.imshow("EndPose Recognizer (實驗)",frame)

    key=cv2.waitKey(1)&0xFF
    if key==ord('q'): break
    elif key==ord('c'):
        word_buffer.clear(); last_top3=[]; last_seq_top3=[]; last_end_top3=[]
        sign_frames=[]; stable_buffer.clear(); print("  已清空")
    elif key==ord(' '):
        paused=not paused; print(f"  {'暫停' if paused else '繼續'}")

cap.release()
cv2.destroyAllWindows()
if word_buffer:
    print(f"\n最終詞彙：{' '.join(word_buffer)}")
