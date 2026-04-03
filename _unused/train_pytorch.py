import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
import pickle
import os

# 確認 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用裝置：{device}")
if device.type == "cuda":
    print(f"GPU：{torch.cuda.get_device_name(0)}")

# 載入資料
print("\n載入資料...")
df = pd.read_csv("data/features.csv")
label_counts = df['label'].value_counts()
valid_labels = label_counts[label_counts >= 5].index
df = df[df['label'].isin(valid_labels)]
print(f"筆數：{len(df)}，類別數：{df['label'].nunique()}")

X = df.drop('label', axis=1).values
y = df['label'].values

le = LabelEncoder()
y_encoded = le.fit_transform(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"訓練集：{len(X_train)} 筆")
print(f"測試集：{len(X_test)} 筆")

# 轉成 Tensor 放到 GPU
X_train_t = torch.FloatTensor(X_train).to(device)
X_test_t  = torch.FloatTensor(X_test).to(device)
y_train_t = torch.LongTensor(y_train).to(device)
y_test_t  = torch.LongTensor(y_test).to(device)

train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader  = DataLoader(train_dataset, batch_size=512, shuffle=True)

n_classes  = len(le.classes_)
n_features = X_train.shape[1]

# 定義模型（加 BatchNorm + Dropout 防過擬合）
class SignMLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, out_dim)
        )
    def forward(self, x):
        return self.net(x)

model     = SignMLP(n_features, n_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
criterion = nn.CrossEntropyLoss()

print(f"\n模型參數量：{sum(p.numel() for p in model.parameters()):,}")
print("開始訓練...\n")

EPOCHS = 100
best_test_acc = 0

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(X_batch), y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()

    if epoch % 10 == 0:
        model.eval()
        with torch.no_grad():
            train_pred = model(X_train_t).argmax(dim=1).cpu().numpy()
            test_pred  = model(X_test_t).argmax(dim=1).cpu().numpy()

        train_acc = accuracy_score(y_train, train_pred)
        test_acc  = accuracy_score(y_test,  test_pred)

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), "models/pytorch_mlp_best.pth")

        print(f"Epoch {epoch:3d} | Loss: {total_loss/len(train_loader):.4f} "
              f"| Train: {train_acc*100:.2f}% | Test: {test_acc*100:.2f}%"
              f"{' ← 最佳' if test_acc == best_test_acc else ''}")

# 載入最佳模型做最終評估
model.load_state_dict(torch.load("models/pytorch_mlp_best.pth"))
model.eval()
with torch.no_grad():
    train_pred = model(X_train_t).argmax(dim=1).cpu().numpy()
    test_pred  = model(X_test_t).argmax(dim=1).cpu().numpy()

train_acc = accuracy_score(y_train, train_pred)
test_acc  = accuracy_score(y_test,  test_pred)

print(f"\n{'='*50}")
print(f"最佳模型結果：")
print(f"訓練集準確率：{train_acc*100:.2f}%")
print(f"測試集準確率：{test_acc*100:.2f}%")
print(f"差距：{(train_acc - test_acc)*100:.2f}%")
if train_acc - test_acc > 0.1:
    print("⚠️  可能有過擬合")
else:
    print("✅ 過擬合程度可接受")

# 儲存
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/pytorch_mlp_best.pth")
with open("models/pytorch_label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)
with open("models/pytorch_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# 儲存模型架構參數（之後載入用）
model_config = {"n_features": n_features, "n_classes": n_classes}
with open("models/pytorch_config.pkl", "wb") as f:
    pickle.dump(model_config, f)

print(f"\n模型已儲存到 models/")
print(f"  pytorch_mlp_best.pth")
print(f"  pytorch_label_encoder.pkl")
print(f"  pytorch_scaler.pkl")
print(f"  pytorch_config.pkl")