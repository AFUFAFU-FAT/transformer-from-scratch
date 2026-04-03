"""
train_lstm.py
台灣手語辨識 BiLSTM 分類器（PyTorch + GPU）

輸入：data/sequences.npz  → X (N, 32, 66)，y (N,)
輸出：models/lstm_best.pth
      models/lstm_config.pkl

架構：
  輸入 (batch, 32, 66)
      ↓
  Input Projection: Linear(66 → 128)  ← 先升維，讓 LSTM 有更多空間學習
      ↓
  BiLSTM Layer 1: 128 → 256 (bidirectional → 實際輸出 512)
      ↓  Dropout 0.4
  BiLSTM Layer 2: 256 → 128 (bidirectional → 實際輸出 256)
      ↓
  取最後一個時間步的隱藏狀態
      ↓
  FC: 256 → 512 → 3495
      ↓
  輸出 logits
"""

import os
import time
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ────────────────────────────────────────────────────────────
#  設定
# ────────────────────────────────────────────────────────────
SEQ_PATH    = "data/sequences.npz"
LABEL_PATH  = "data/seq_label_encoder.pkl"
MODEL_DIR   = "models"
MODEL_PATH  = os.path.join(MODEL_DIR, "lstm_best.pth")
CONFIG_PATH = os.path.join(MODEL_DIR, "lstm_config.pkl")

BATCH_SIZE   = 256
EPOCHS       = 80
LR           = 1e-4
WEIGHT_DECAY = 1e-3
DROPOUT      = 0.6
PATIENCE     = 15

# ────────────────────────────────────────────────────────────
#  裝置
# ────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  裝置：{device}")
if device.type == "cuda":
    print(f"    GPU：{torch.cuda.get_device_name(0)}")

# ────────────────────────────────────────────────────────────
#  載入資料（build_sequences.py 已預先切分，直接讀取）
# ────────────────────────────────────────────────────────────
print("\n📂 載入序列資料...")
data = np.load(SEQ_PATH)
X_train_raw = data["X_train"]
y_train     = data["y_train"]
X_test_raw  = data["X_test"]
y_test      = data["y_test"]

with open(LABEL_PATH, "rb") as f:
    le = pickle.load(f)

num_classes = len(le.classes_)
seq_len     = X_train_raw.shape[1]
input_dim   = X_train_raw.shape[2]

print(f"   訓練序列：{len(X_train_raw):,}  測試序列：{len(X_test_raw):,}")
print(f"   Shape：{X_train_raw.shape}  →  (N, frames, features)")
print(f"   類別數：{num_classes}")
print(f"   測試集涵蓋類別：{len(np.unique(y_test))}/{num_classes}")

# ────────────────────────────────────────────────────────────
#  特徵正規化（僅用訓練集計算 mean/std，避免洩漏）
# ────────────────────────────────────────────────────────────
X_flat    = X_train_raw.reshape(-1, input_dim)
feat_mean = X_flat.mean(axis=0).astype(np.float32)
feat_std  = X_flat.std(axis=0).astype(np.float32) + 1e-8

X_train = (X_train_raw - feat_mean) / feat_std
X_test  = (X_test_raw  - feat_mean) / feat_std

X_train_t = torch.FloatTensor(X_train)
X_test_t  = torch.FloatTensor(X_test)
y_train_t = torch.LongTensor(y_train)
y_test_t  = torch.LongTensor(y_test)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t),
                          batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=0, pin_memory=True)
test_loader  = DataLoader(TensorDataset(X_test_t, y_test_t),
                          batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=0, pin_memory=True)

# ────────────────────────────────────────────────────────────
#  模型
# ────────────────────────────────────────────────────────────
class SignLanguageBiLSTM(nn.Module):
    """
    手語序列辨識 BiLSTM
    
    輸入：(batch, seq_len, input_dim)  e.g. (512, 32, 66)
    輸出：(batch, num_classes)         e.g. (512, 3495)
    """
    def __init__(self, input_dim, num_classes, hidden_dim=64, num_layers=1, dropout=0.4):
        super().__init__()

        # 輸入升維
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
        )

        # BiLSTM（bidirectional=True → 輸出維度 × 2）
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )

        lstm_out_dim = hidden_dim * 2  # bidirectional

        # 注意力加權池化（比單純取最後一幀更好）
        self.attention = nn.Linear(lstm_out_dim, 1)

        # 分類頭（直接映射，不加中間層）
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_out_dim, num_classes),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)

        # 輸入投影
        x = self.input_proj(x)                    # (batch, seq_len, 128)

        # LSTM
        lstm_out, _ = self.lstm(x)                # (batch, seq_len, hidden*2)
        lstm_out = self.dropout(lstm_out)

        # 注意力加權池化：讓模型自己學哪幾幀最重要
        attn_weights = torch.softmax(
            self.attention(lstm_out), dim=1
        )                                          # (batch, seq_len, 1)
        context = (attn_weights * lstm_out).sum(dim=1)  # (batch, hidden*2)

        # 分類
        logits = self.classifier(context)          # (batch, num_classes)
        return logits


model = SignLanguageBiLSTM(
    input_dim=input_dim,
    num_classes=num_classes,
    hidden_dim=64,
    num_layers=1,
    dropout=DROPOUT,
).to(device)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n🧠 模型架構：BiLSTM + Attention Pooling")
print(f"   可訓練參數：{total_params:,}")

# ────────────────────────────────────────────────────────────
#  訓練設定
# ────────────────────────────────────────────────────────────
criterion = nn.CrossEntropyLoss(label_smoothing=0.15)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=20, T_mult=2, eta_min=1e-5
)

# ────────────────────────────────────────────────────────────
#  訓練 / 評估函數
# ────────────────────────────────────────────────────────────
def train_epoch(model, loader):
    model.train()
    total_loss, total_correct, total = 0.0, 0, 0
    for X_b, y_b in loader:
        X_b, y_b = X_b.to(device, non_blocking=True), y_b.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(X_b)
        loss = criterion(logits, y_b)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss    += loss.item() * len(y_b)
        total_correct += (logits.argmax(1) == y_b).sum().item()
        total         += len(y_b)
    return total_loss / total, total_correct / total


@torch.no_grad()
def eval_epoch(model, loader):
    model.eval()
    total_loss, total_correct, total = 0.0, 0, 0
    for X_b, y_b in loader:
        X_b, y_b = X_b.to(device, non_blocking=True), y_b.to(device, non_blocking=True)
        logits = model(X_b)
        loss = criterion(logits, y_b)
        total_loss    += loss.item() * len(y_b)
        total_correct += (logits.argmax(1) == y_b).sum().item()
        total         += len(y_b)
    return total_loss / total, total_correct / total

# ────────────────────────────────────────────────────────────
#  主訓練迴圈
# ────────────────────────────────────────────────────────────
print(f"\n🚀 開始訓練  Epochs={EPOCHS}  Batch={BATCH_SIZE}  LR={LR}")
print("-" * 72)

best_val_acc  = 0.0
best_epoch    = 0
patience_cnt  = 0
history       = []

os.makedirs(MODEL_DIR, exist_ok=True)

for epoch in range(1, EPOCHS + 1):
    t0 = time.time()
    train_loss, train_acc = train_epoch(model, train_loader)
    val_loss,   val_acc   = eval_epoch(model,  test_loader)
    scheduler.step()

    elapsed  = time.time() - t0
    gap      = train_acc - val_acc
    lr_now   = scheduler.get_last_lr()[0]

    history.append(dict(epoch=epoch,
                        train_acc=train_acc, val_acc=val_acc, gap=gap))

    marker = ""
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch   = epoch
        patience_cnt = 0
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "val_acc": val_acc,
            "num_classes": num_classes,
            "input_dim": input_dim,
            "seq_len": seq_len,
            "feat_mean": feat_mean,
            "feat_std":  feat_std,
        }, MODEL_PATH)
        marker = " ← 最佳"
    else:
        patience_cnt += 1

    if epoch % 5 == 0 or epoch == 1 or marker:
        print(f"Epoch {epoch:3d}/{EPOCHS} | "
              f"Train {train_acc*100:.2f}% | "
              f"Val {val_acc*100:.2f}% | "
              f"Gap {gap*100:.2f}% | "
              f"LR {lr_now:.1e} | {elapsed:.1f}s{marker}")

    if patience_cnt >= PATIENCE:
        print(f"\n⏹️  Early Stopping（Epoch {epoch}，連續 {PATIENCE} 輪無改善）")
        break

# ────────────────────────────────────────────────────────────
#  儲存 config（推論時需要）
# ────────────────────────────────────────────────────────────
config = {
    "input_dim":   input_dim,
    "seq_len":     seq_len,
    "num_classes": num_classes,
    "hidden_dim":  64,
    "num_layers":  1,
    "dropout":     DROPOUT,
    "feat_mean":   feat_mean,
    "feat_std":    feat_std,
}
with open(CONFIG_PATH, "wb") as f:
    pickle.dump(config, f)

# ────────────────────────────────────────────────────────────
#  最終結果
# ────────────────────────────────────────────────────────────
print(f"\n{'='*72}")
print(f"📊 訓練完成")
print(f"   最佳驗證準確率：{best_val_acc*100:.2f}%  (Epoch {best_epoch})")

best_row = next(r for r in history if r["epoch"] == best_epoch)
print(f"   訓練準確率：    {best_row['train_acc']*100:.2f}%")
print(f"   Train-Val 差距：{best_row['gap']*100:.2f}%")

print(f"\n🆚 模型對比")
print(f"   MLP（單幀）  88.12%")
print(f"   BiLSTM（序列）{best_val_acc*100:.2f}%")

if best_row['gap'] < 0.05:
    print("   ✅ 過擬合控制良好")
elif best_row['gap'] < 0.10:
    print("   ⚠️  輕微過擬合")
else:
    print("   ❌ 過擬合需改善（可嘗試增大 dropout 或減少 hidden_dim）")

print(f"\n   模型：{MODEL_PATH}")
print(f"   Config：{CONFIG_PATH}")
print(f"\n✅ 可繼續執行 recognize.py 進行即時辨識測試")

# ────────────────────────────────────────────────────────────
#  Confusion Matrix（載入最佳模型跑測試集）
# ────────────────────────────────────────────────────────────
print(f"\n{'='*72}")
print(f"📊 Confusion Matrix 分析（測試集）")

best_ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
model.load_state_dict(best_ckpt["model_state_dict"])
model.eval()

all_preds, all_true = [], []
with torch.no_grad():
    for X_b, y_b in test_loader:
        X_b = X_b.to(device, non_blocking=True)
        logits = model(X_b)
        preds = logits.argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_true.extend(y_b.numpy())

all_preds = np.array(all_preds)
all_true  = np.array(all_true)

cm = confusion_matrix(all_true, all_preds)

# ── Top 20 最容易混淆的詞彙對 ──
confusion_pairs = []
n = cm.shape[0]
for i in range(n):
    for j in range(n):
        if i != j and cm[i, j] > 0:
            true_label = le.inverse_transform([i])[0]
            pred_label = le.inverse_transform([j])[0]
            confusion_pairs.append((cm[i, j], true_label, pred_label))

confusion_pairs.sort(reverse=True)

print(f"\n   Top 20 最容易混淆的詞彙對（真實→預測）：")
print(f"   {'次數':>4}  {'真實':>8} → {'預測'}")
print(f"   {'-'*40}")
for count, true_lbl, pred_lbl in confusion_pairs[:20]:
    print(f"   {count:>4}  {true_lbl:>8} → {pred_lbl}")

# ── 各詞彙召回率（找出最差的詞彙）──
recall_per_class = []
for i in range(n):
    total = cm[i].sum()
    correct = cm[i, i]
    recall = correct / total if total > 0 else 0.0
    recall_per_class.append((recall, le.inverse_transform([i])[0], int(total)))

recall_per_class.sort()
print(f"\n   召回率最低的 15 個詞彙：")
print(f"   {'召回率':>6}  {'詞彙':>10}  {'測試樣本數':>8}")
print(f"   {'-'*35}")
for recall, lbl, total in recall_per_class[:15]:
    print(f"   {recall*100:>5.1f}%  {lbl:>10}  {total:>8}")
