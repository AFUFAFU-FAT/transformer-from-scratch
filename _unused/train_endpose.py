"""
train_endpose.py
用「結尾單幀」訓練 MLP 分類器

策略：
  每幀特徵獨立分類，不依賴時間序列
  特徵 179 維（與 build_sequences.py 一致：167 CSV + 2 朝向 + 10 指伸展）
  適合「手停下來那一幀」的辨識場景
"""

import numpy as np
import pandas as pd
import pickle
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder

# ────────────────────────────────────────────────────────────
#  設定
# ────────────────────────────────────────────────────────────
DATA_PATH   = "data/features_filtered.csv"
MODEL_DIR   = "models"
MODEL_PATH  = os.path.join(MODEL_DIR, "mlp_endpose.pth")
CONFIG_PATH = os.path.join(MODEL_DIR, "mlp_endpose_config.pkl")

CSV_FEAT_DIM = 167
BATCH_SIZE   = 512
EPOCHS       = 80
LR           = 1e-3
WEIGHT_DECAY = 1e-3
DROPOUT      = 0.4
PATIENCE     = 15

TIP_IDX = [4,  8, 12, 16, 20]
MCP_IDX = [1,  5,  9, 13, 17]

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"裝置：{device}")

# ────────────────────────────────────────────────────────────
#  特徵工程（與 build_sequences.py 一致）
# ────────────────────────────────────────────────────────────
def compute_finger_extensions(lm63_batch):
    lms   = lm63_batch.reshape(-1, 21, 3)
    wrist = lms[:, 0:1, :]
    tips  = lms[:, TIP_IDX, :]
    mcps  = lms[:, MCP_IDX, :]
    tip_d = np.linalg.norm(tips - wrist, axis=2)
    mcp_d = np.linalg.norm(mcps - wrist, axis=2) + 1e-6
    return (tip_d / mcp_d).astype(np.float32)


def apply_feature_weights(frames):
    out = frames.copy()
    for offset in (0, 68):
        wx = frames[:, offset + 0]
        wy = frames[:, offset + 1]
        mx = frames[:, offset + 27]
        my = frames[:, offset + 28]
        scale = np.sqrt((mx - wx) ** 2 + (my - wy) ** 2) + 1e-6
        for i in range(21):
            b = offset + i * 3
            w = W_FINGERTIP if i in FINGERTIPS else W_FINGER
            out[:, b + 0] = (frames[:, b + 0] - wx) / scale * w
            out[:, b + 1] = (frames[:, b + 1] - wy) / scale * w
            out[:, b + 2] = frames[:, b + 2] * w
        out[:, offset + 66] = frames[:, offset + 66] * W_PALM_DIR
        out[:, offset + 67] = frames[:, offset + 67] * W_PALM_DIR
        out[:, offset + 63] = frames[:, offset + 63] * W_POSITION
        out[:, offset + 64] = frames[:, offset + 64] * W_POSITION
        out[:, offset + 65] = frames[:, offset + 65] * W_POSITION
    out[:, 136:146] = frames[:, 136:146] * W_ANCHOR
    out[:, 149:161] = frames[:, 149:161] * W_ARM
    out[:, 161:167] = frames[:, 161:167] * W_FACE
    out[:, 167:169] = frames[:, 167:169] * W_ORIENT
    out[:, 169:179] = frames[:, 169:179] * W_HANDSHAPE
    return out


def process_features(raw):
    l_sho = raw[:, 138:140]
    r_sho = raw[:, 140:142]
    sho_vec = r_sho - l_sho
    theta = np.arctan2(sho_vec[:, 1], sho_vec[:, 0])
    body_orient = np.stack([np.cos(theta), np.sin(theta)], axis=1).astype(np.float32)
    ext_r = compute_finger_extensions(raw[:, :63])
    ext_l = compute_finger_extensions(raw[:, 68:131])
    frames = np.concatenate([raw, body_orient, ext_r, ext_l], axis=1)
    return apply_feature_weights(frames)

# ────────────────────────────────────────────────────────────
#  載入資料
# ────────────────────────────────────────────────────────────
print("📂 載入資料...")
df = pd.read_csv(DATA_PATH)
feature_cols = [f"lm_{i}" for i in range(CSV_FEAT_DIM)]
for col in feature_cols:
    if col not in df.columns:
        df[col] = 0.0
df = df.dropna(subset=feature_cols)
print(f"   {len(df):,} 筆  {df['label'].nunique()} 個詞彙")

raw = df[feature_cols].values.astype(np.float32)
labels = df["label"].values

X = process_features(raw)  # (N, 179)

le = LabelEncoder()
y = le.fit_transform(labels)
num_classes = len(le.classes_)
input_dim = X.shape[1]
print(f"   特徵維度：{input_dim}  類別數：{num_classes}")

# 時間切分（前 80% 訓練，後 20% 驗證）
n_train = int(len(X) * 0.8)
X_train, X_val = X[:n_train], X[n_train:]
y_train, y_val = y[:n_train], y[n_train:]

feat_mean = X_train.mean(axis=0).astype(np.float32)
feat_std  = X_train.std(axis=0).astype(np.float32) + 1e-8
X_train = (X_train - feat_mean) / feat_std
X_val   = (X_val   - feat_mean) / feat_std

train_loader = DataLoader(
    TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train)),
    batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(
    TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val)),
    batch_size=BATCH_SIZE, shuffle=False)

# ────────────────────────────────────────────────────────────
#  模型
# ────────────────────────────────────────────────────────────
class EndposeMLP(nn.Module):
    def __init__(self, input_dim, num_classes, dropout=0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )
    def forward(self, x):
        return self.net(x)

model = EndposeMLP(input_dim, num_classes, DROPOUT).to(device)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n🧠 EndposeMLP  可訓練參數：{total_params:,}")

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, eta_min=1e-5)

# ────────────────────────────────────────────────────────────
#  訓練
# ────────────────────────────────────────────────────────────
best_val, best_state, no_improve = 0.0, None, 0

print(f"\n開始訓練  Epochs={EPOCHS}  Batch={BATCH_SIZE}")
print("-" * 60)

for epoch in range(1, EPOCHS + 1):
    model.train()
    correct, total = 0, 0
    for X_b, y_b in train_loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(X_b), y_b)
        loss.backward()
        optimizer.step()
        preds = model(X_b).argmax(1)
        correct += (preds == y_b).sum().item()
        total += len(y_b)
    train_acc = correct / total

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X_b, y_b in val_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            preds = model(X_b).argmax(1)
            correct += (preds == y_b).sum().item()
            total += len(y_b)
    val_acc = correct / total
    scheduler.step()

    marker = ""
    if val_acc > best_val:
        best_val = val_acc
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        no_improve = 0
        marker = " ← 最佳"
    else:
        no_improve += 1

    if epoch % 5 == 0 or marker:
        print(f"Epoch {epoch:3d}/{EPOCHS} | Train {train_acc*100:.2f}% | Val {val_acc*100:.2f}%"
              f" | Gap {(train_acc-val_acc)*100:.2f}%{marker}")

    if no_improve >= PATIENCE:
        print(f"\n⏹️  Early Stopping（Epoch {epoch}，連續 {PATIENCE} 輪無改善）")
        break

# ────────────────────────────────────────────────────────────
#  儲存
# ────────────────────────────────────────────────────────────
os.makedirs(MODEL_DIR, exist_ok=True)
torch.save({"model_state_dict": best_state, "val_acc": best_val}, MODEL_PATH)

config = {
    "input_dim":  input_dim,
    "num_classes": num_classes,
    "dropout":    DROPOUT,
    "feat_mean":  feat_mean,
    "feat_std":   feat_std,
}
with open(CONFIG_PATH, "wb") as f:
    pickle.dump(config, f)
with open("models/mlp_endpose_label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

print(f"\n{'='*60}")
print(f"📊 訓練完成")
print(f"   最佳驗證準確率：{best_val*100:.2f}%")
print(f"   模型：{MODEL_PATH}")
print(f"\n接著執行：python recognize_endpose.py")
