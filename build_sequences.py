"""
build_sequences.py（v2）
改用滑動窗口 + 資料增強，大幅增加序列數量解決過擬合

策略：
  stride=4  → 100幀詞彙切出 ~17 個序列（原本只有 3 個）
  + 隨機時間縮放抖動（±20%）
  + 高斯雜訊注入
"""

import numpy as np
import pandas as pd
import pickle
import os
from sklearn.preprocessing import LabelEncoder

DATA_PATH      = "data/features_filtered.csv"
TEST_DATA_PATH = "data/test_features.csv"    # 若此檔存在，用獨立測試集（不做時間切分）
SEQ_OUT_PATH   = "data/sequences.npz"
LABEL_PATH     = "data/seq_label_encoder.pkl"

CSV_FEAT_DIM = 167   # CSV 原始維度：149 + 雙臂特徵 12 + 手臉距離 6
FEAT_DIM     = 199   # 167 + 身體朝向2 + 伸展分數10 + 伸展二值10 + 食指方向6 + 捏合距離2 + 手部偵測旗標2

# ── 手指關節索引 ──────────────────────────────────────────────
TIP_IDX = [4,  8, 12, 16, 20]   # 拇指到小指指尖
MCP_IDX = [1,  5,  9, 13, 17]   # 拇指到小指指節（MCP）
# 關節角度判斷：量測 PIP 關節的彎曲角度（>130° = 伸直，否則彎曲）
FINGER_JOINTS = [(2,3,4), (5,6,7), (9,10,11), (13,14,15), (17,18,19)]
EXTEND_ANGLE  = 140.0

# ── 特徵權重 ──────────────────────────────────────────────────
# 優先順序：手形 > 掌向 > 位置（台灣手語 5 大要素，手形最關鍵）
# W_HANDSHAPE 最高：明確編碼哪根手指伸出，直接區分「甚麼」(食指)vs開掌
W_HANDSHAPE = 6.0   # 手指伸展分數（連續）→ 手形
W_BINARY    = 10.0  # 手指伸展二值（哪隻伸出）→ 手語最核心判斷基準
W_POINTING  = 8.0   # 食指指向方向（你/我/說/他/我們 都是[0,1,0,0,0]，靠方向區分）
W_FINGERTIP = 4.0   # 指尖座標（相對手腕正規化）→ 手形輔助
W_FINGER    = 3.0   # 其他手部關節 → 手形
W_PALM_DIR  = 3.0   # 掌心方向（orientation）
W_POSITION  = 4.0   # 手腕相對身體位置（rel_wrist，高度直接區分爸爸/哥哥等混淆對）
W_ARM       = 2.5   # 手臂特徵（手肘位置 + 上臂/前臂方向）→ 手臂動作
W_FACE      = 5.0   # 手臉距離（手腕到鼻子/嘴/耳朵）→ 區分貼臉vs非貼臉手勢
W_ORIENT    = 2.0   # 身體朝向（肩膀向量的 cos/sin）→ 讓座標相對身體面向不變
W_ANCHOR    = 1.5   # 身體錨點 → 位置（稍降，手形優先）

FINGERTIPS = {4, 8, 12, 16, 20}

SEQ_LEN   = 32
STRIDE    = 16     # 滑動步長（原4太小，導致87%重複序列，模型直接記憶）
MIN_FRAMES = 16    # 少於此幀數的詞彙跳過


def compute_finger_extensions(lm63_batch):
    """
    從 63 維手部關鍵點批次計算 5 指伸展分數。
    lm63_batch: shape (N, 63) — 21 landmarks × 3 (x,y,z)
    返回: shape (N, 5) — 每指 tip-to-wrist / mcp-to-wrist 比值
    """
    lms    = lm63_batch.reshape(-1, 21, 3)
    wrist  = lms[:, 0:1, :]
    tips   = lms[:, TIP_IDX, :]
    mcps   = lms[:, MCP_IDX, :]
    tip_d  = np.linalg.norm(tips - wrist, axis=2)
    mcp_d  = np.linalg.norm(mcps - wrist, axis=2) + 1e-6
    return (tip_d / mcp_d).astype(np.float32)


def compute_finger_binary(lm63_batch):
    """
    從 63 維手部關鍵點批次計算 5 指伸展二值（純角度判斷，無遮蔽估計）。
    量測 PIP/IP 關節角度：> 130° → 伸出(1)，否則彎曲(0)
    例：認真=[1,1,1,1,1]，找(OK)=[0,0,1,1,1]，握拳=[0,0,0,0,0]
    """
    lms    = lm63_batch.reshape(-1, 21, 3)
    binary = np.zeros((len(lms), 5), dtype=np.float32)
    for fi, (a, b, c) in enumerate(FINGER_JOINTS):
        v1    = lms[:, a, :] - lms[:, b, :]
        v2    = lms[:, c, :] - lms[:, b, :]
        cos_a = np.sum(v1*v2, axis=1) / (
                    np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-6)
        binary[:, fi] = (np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))) > EXTEND_ANGLE)
    return binary.astype(np.float32)


def compute_pinch_dist(lm63_batch):
    """
    捏合距離：||拇指尖 - 食指尖|| / 掌長（正規化）
    找(OK)：拇指食指碰觸 → 小值(≈0)
    再見/認真：五指展開 → 大值(≈1.5~2.5)
    lm63_batch: (N, 63)  →  (N, 1)
    """
    lms       = lm63_batch.reshape(-1, 21, 3)
    thumb_tip = lms[:, 4, :]
    index_tip = lms[:, 8, :]
    hand_size = np.linalg.norm(lms[:, 9, :] - lms[:, 0, :], axis=1) + 1e-6  # 掌長
    dist      = np.linalg.norm(thumb_tip - index_tip, axis=1) / hand_size
    return np.minimum(dist, 3.0).reshape(-1, 1).astype(np.float32)


def compute_pointing_dir(lm63_batch):
    """
    食指指向方向（3D 正規化向量）：normalize(tip - MCP)
    用於區分同樣是 binary=[0,1,0,0,0] 的手勢：
      你（向前）、我（向自己）、說（從嘴向前）、他（向右）、我們（繞圈）
    lm63_batch: (N, 63)  →  (N, 3)
    """
    lms = lm63_batch.reshape(-1, 21, 3)
    direction = lms[:, 8, :] - lms[:, 5, :]   # index tip - index MCP
    norms = np.linalg.norm(direction, axis=1, keepdims=True) + 1e-6
    return (direction / norms).astype(np.float32)


def apply_feature_weights(frames):
    """
    將 149 維特徵重新加權：手指形狀 > 位置 > 身體錨點

    每隻手的 63 維關鍵點先轉為「相對手腕、以掌長正規化」的純手形座標，
    再乘上對應權重。兩手各自處理，缺手（全零）不影響結果。

    輸入/輸出 shape: (N, 149)
    """
    out = frames.copy()

    for offset in (0, 68):          # 右手 offset=0，左手 offset=68
        wx = frames[:, offset + 0]  # wrist x
        wy = frames[:, offset + 1]  # wrist y
        # landmark 9 = 中指根部，用來估計掌長
        mx = frames[:, offset + 27]
        my = frames[:, offset + 28]
        scale = np.sqrt((mx - wx) ** 2 + (my - wy) ** 2) + 1e-6

        for i in range(21):          # 21 個關鍵點
            b = offset + i * 3
            w = W_FINGERTIP if i in FINGERTIPS else W_FINGER
            out[:, b + 0] = (frames[:, b + 0] - wx) / scale * w
            out[:, b + 1] = (frames[:, b + 1] - wy) / scale * w
            out[:, b + 2] = frames[:, b + 2] * w

        # 手掌朝向 (palm_dir) = offset+66, offset+67
        out[:, offset + 66] = frames[:, offset + 66] * W_PALM_DIR
        out[:, offset + 67] = frames[:, offset + 67] * W_PALM_DIR

        # 相對手腕位置 (rel_wrist) = offset+63..65 → 高度區分頭頂/胸前/腰部
        out[:, offset + 63] = frames[:, offset + 63] * W_POSITION
        out[:, offset + 64] = frames[:, offset + 64] * W_POSITION
        out[:, offset + 65] = frames[:, offset + 65] * W_POSITION

    # 身體錨點 lm_136..145
    out[:, 136:146] = frames[:, 136:146] * W_ANCHOR

    # 手臂特徵 149..160（右臂6 + 左臂6）
    out[:, 149:161] = frames[:, 149:161] * W_ARM

    # 手臉距離 161..166（右手3 + 左手3）
    out[:, 161:167] = frames[:, 161:167] * W_FACE

    # 身體朝向 167..168（cos θ, sin θ of 肩膀向量）
    out[:, 167:169] = frames[:, 167:169] * W_ORIENT

    # 手指伸展分數 169..178（右手5 + 左手5）
    out[:, 169:179] = frames[:, 169:179] * W_HANDSHAPE

    # 手指伸展二值 179..188（右手5 + 左手5）— 1=伸出, 0=彎曲
    out[:, 179:189] = frames[:, 179:189] * W_BINARY

    # 食指指向方向 189..194（右手 dx/dy/dz + 左手 dx/dy/dz）
    out[:, 189:195] = frames[:, 189:195] * W_POINTING

    # 捏合距離 195..196（右手, 左手）— 找(OK)≈0, 再見/認真≈1.5~2.5
    out[:, 195:197] = frames[:, 195:197] * W_BINARY   # 同樣高權重

    # 手部偵測旗標 197..198（is_right, is_left）— 單手 vs 雙手詞彙最直接判斷
    out[:, 197:199] = frames[:, 197:199] * W_BINARY

    return out


def uniform_sample(frames, target_len):
    T = len(frames)
    if T == target_len:
        return frames.copy()
    indices = np.linspace(0, T - 1, target_len, dtype=int)
    return frames[indices]


def sliding_window(frames, seq_len, stride):
    """用滑動窗口從一段幀序列切出多個子序列"""
    seqs = []
    T = len(frames)
    if T < seq_len:
        # 幀數不夠：loop padding 後當一個序列
        repeats = int(np.ceil(seq_len / T))
        padded = np.tile(frames, (repeats, 1))[:seq_len]
        seqs.append(padded)
    else:
        for start in range(0, T - seq_len + 1, stride):
            seqs.append(frames[start:start + seq_len].copy())
        # 確保最後一段也被納入
        if (T - seq_len) % stride != 0:
            seqs.append(frames[T - seq_len:].copy())
    return seqs


def augment_sequence(seq, noise_std=0.005):
    """高斯雜訊增強：前 126 維加微小雜訊"""
    augmented = seq.copy()
    augmented[:, :126] += np.random.normal(0, noise_std, (len(seq), 126)).astype(np.float32)
    return augmented


def augment_with_random_crop(frames, seq_len, n_crops=5, hard=False):
    """
    隨機起點增強：模擬即時辨識截到動作中間段的情況。
    hard=True（測試集用）：
      - 最短截取長度降到 seq_len//4（模擬只截到四分之一動作）
      - 加入隨機時間扭曲（±30% 速度變化）
    hard=False（訓練集用）：
      - 最短截取長度 seq_len//3
    """
    T = len(frames)
    min_crop = seq_len // 4 if hard else seq_len // 3
    seqs = []
    for _ in range(n_crops):
        crop_len = np.random.randint(min_crop, seq_len + 1)
        start    = np.random.randint(0, max(0, T - crop_len) + 1)
        seg      = frames[start:start + crop_len].copy()

        # 時間扭曲（hard 模式：隨機加速或減速）
        if hard and len(seg) >= 4:
            speed = np.random.uniform(0.7, 1.4)
            new_len = max(4, int(len(seg) * speed))
            indices = np.linspace(0, len(seg) - 1, new_len, dtype=int)
            seg = seg[indices]

        if len(seg) < seq_len:
            pad = np.tile(seg[-1], (seq_len - len(seg), 1))
            seg = np.vstack([seg, pad])
        seqs.append(seg[:seq_len])
    return seqs


print(f"📂 載入 {DATA_PATH} ...")
df_raw = pd.read_csv(DATA_PATH)
print(f"   原始筆數：{len(df_raw):,}  詞彙數：{df_raw['label'].nunique()}")

feature_cols = [f"lm_{i}" for i in range(CSV_FEAT_DIM)]
# 舊格式（149維）自動補零手臂特徵，保持向下相容
for col in feature_cols:
    if col not in df_raw.columns:
        df_raw[col] = 0.0
df = df_raw.dropna(subset=feature_cols)
dropped = len(df_raw) - len(df)
if dropped:
    print(f"   ⚠️  移除 {dropped} 筆舊格式（81維）資料")
print(f"   有效筆數：{len(df):,}  詞彙數：{df['label'].nunique()}")
print(f"   特徵權重：手指×{W_FINGER}  掌向×{W_PALM_DIR}  位置×{W_POSITION}  錨點×{W_ANCHOR}")

# 訓練集的詞彙白名單（以 features_filtered.csv 的標籤為準）
ALLOWED_LABELS = set(df['label'].unique())

# ── 判斷是否有獨立測試集 ──────────────────────────────────────
use_external_test = os.path.exists(TEST_DATA_PATH)
if use_external_test:
    df_test_raw = pd.read_csv(TEST_DATA_PATH)
    for col in feature_cols:
        if col not in df_test_raw.columns:
            df_test_raw[col] = 0.0
    df_test_raw = df_test_raw[df_test_raw['label'].isin(ALLOWED_LABELS)]  # 與訓練集同步
    df_test     = df_test_raw.dropna(subset=feature_cols)
    print(f"\n✅ 找到獨立測試集：{TEST_DATA_PATH}")
    print(f"   測試筆數：{len(df_test):,}  詞彙數：{df_test['label'].nunique()}")
else:
    print(f"\n⚠️  未找到 {TEST_DATA_PATH}，使用時間切分（訓練集後 20%）")

train_X, train_y = [], []
test_X,  test_y  = [], []
stats = {"total_words": 0, "skipped": 0}

print(f"\n🔧 建立序列（SEQ_LEN={SEQ_LEN}, STRIDE={STRIDE}）")

def process_features(raw):
    # 身體朝向：從肩膀座標（body_anchors 中的 l_sho/r_sho）推導
    # body_anchors 在 136:146：nose(2) l_sho(2) r_sho(2) l_hip(2) r_hip(2)
    l_sho = raw[:, 138:140]   # left shoulder (x, y)
    r_sho = raw[:, 140:142]   # right shoulder (x, y)
    sho_vec = r_sho - l_sho
    theta = np.arctan2(sho_vec[:, 1], sho_vec[:, 0])
    body_orient = np.stack([np.cos(theta), np.sin(theta)], axis=1).astype(np.float32)

    ext_r  = compute_finger_extensions(raw[:, :63])
    ext_l  = compute_finger_extensions(raw[:, 68:131])
    bin_r   = compute_finger_binary(raw[:, :63])
    bin_l   = compute_finger_binary(raw[:, 68:131])
    ptr_r   = compute_pointing_dir(raw[:, :63])
    ptr_l   = compute_pointing_dir(raw[:, 68:131])

    # ── 時序平滑（window=3），去除 MediaPipe 跳幀噪音 ──
    if len(bin_r) >= 3:
        kernel = np.array([1/3, 1/3, 1/3])
        for i in range(5):
            bin_r[:, i] = np.convolve(bin_r[:, i], kernel, mode='same')
            bin_l[:, i] = np.convolve(bin_l[:, i], kernel, mode='same')
        for i in range(3):
            ptr_r[:, i] = np.convolve(ptr_r[:, i], kernel, mode='same')
            ptr_l[:, i] = np.convolve(ptr_l[:, i], kernel, mode='same')
        norms_r = np.linalg.norm(ptr_r, axis=1, keepdims=True)
        norms_l = np.linalg.norm(ptr_l, axis=1, keepdims=True)
        ptr_r = ptr_r / (norms_r + 1e-6)
        ptr_l = ptr_l / (norms_l + 1e-6)
    pinch_r = compute_pinch_dist(raw[:, :63])        # (N,1) 右手捏合距離
    pinch_l = compute_pinch_dist(raw[:, 68:131])     # (N,1) 左手捏合距離
    # 手部偵測旗標：右手/左手有無被偵測到（座標全零 = 未偵測）
    is_right = (np.abs(raw[:, :63]).sum(axis=1) > 1e-3).astype(np.float32).reshape(-1, 1)
    is_left  = (np.abs(raw[:, 68:131]).sum(axis=1) > 1e-3).astype(np.float32).reshape(-1, 1)
    frames  = np.concatenate(
        [raw, body_orient, ext_r, ext_l, bin_r, bin_l, ptr_r, ptr_l, pinch_r, pinch_l, is_right, is_left], axis=1)
    return apply_feature_weights(frames)

# ── 訓練集：從 recorded_features.csv ──────────────────────────
for label, group in df.groupby("label", sort=False):
    raw    = group[feature_cols].values.astype(np.float32)
    frames = process_features(raw)
    stats["total_words"] += 1

    if len(frames) < MIN_FRAMES:
        stats["skipped"] += 1
        continue

    if use_external_test:
        # 獨立測試集模式：全部幀都用於訓練
        tr_frames = frames
    else:
        # 時間切分模式：前 80% 訓練，後 20% 測試
        n_train   = max(int(len(frames) * 0.8), MIN_FRAMES)
        n_train   = min(n_train, len(frames) - 1)
        tr_frames = frames[:n_train]
        te_frames = frames[n_train:]
        if len(te_frames) >= SEQ_LEN // 4:
            for seq in augment_with_random_crop(te_frames, SEQ_LEN, n_crops=10, hard=True):
                test_X.append(seq); test_y.append(label)

    for seq in sliding_window(tr_frames, SEQ_LEN, STRIDE):
        train_X.append(seq);                   train_y.append(label)
        train_X.append(augment_sequence(seq)); train_y.append(label)
    for seq in augment_with_random_crop(tr_frames, SEQ_LEN, n_crops=5):
        train_X.append(seq);                   train_y.append(label)

# ── 測試集：從獨立 test_features.csv（若存在）─────────────────
if use_external_test:
    print(f"   建立獨立測試集序列（hard crop×10）...")
    for label, group in df_test.groupby("label", sort=False):
        raw    = group[feature_cols].values.astype(np.float32)
        frames = process_features(raw)
        if len(frames) >= SEQ_LEN // 4:
            for seq in augment_with_random_crop(frames, SEQ_LEN, n_crops=10, hard=True):
                test_X.append(seq); test_y.append(label)

X_train = np.array(train_X, dtype=np.float32)
X_test  = np.array(test_X,  dtype=np.float32)
y_train_raw = np.array(train_y)
y_test_raw  = np.array(test_y)

print(f"\n   詞彙總數：{stats['total_words']}  跳過：{stats['skipped']}")
print(f"   訓練序列：{len(X_train):,}  測試序列：{len(X_test):,}")
print(f"   X_train shape：{X_train.shape}")
print(f"   X_test  shape：{X_test.shape}")

# LabelEncoder 用全部標籤 fit（確保訓練/測試索引一致）
le = LabelEncoder()
le.fit(np.concatenate([y_train_raw, y_test_raw]))
y_train = le.transform(y_train_raw)
y_test  = le.transform(y_test_raw)

os.makedirs("data", exist_ok=True)
np.savez_compressed(SEQ_OUT_PATH,
                    X_train=X_train, y_train=y_train,
                    X_test=X_test,   y_test=y_test)

with open(LABEL_PATH, "wb") as f:
    pickle.dump(le, f)

print(f"\n✅ 儲存完成")
print(f"   {SEQ_OUT_PATH}  →  X_train/y_train/X_test/y_test")
print(f"   {LABEL_PATH}")
print(f"\n接著執行：python train_lstm.py")