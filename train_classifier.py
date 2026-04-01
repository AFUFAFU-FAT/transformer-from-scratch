import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import pickle
import os

print("載入資料...")
df = pd.read_csv("data/features.csv")
print(f"總筆數：{len(df)}")
print(f"詞彙類別數：{df['label'].nunique()}")

# 過濾掉資料量太少的類別（少於 5 筆）
label_counts = df['label'].value_counts()
valid_labels = label_counts[label_counts >= 5].index
df = df[df['label'].isin(valid_labels)]
print(f"過濾後筆數：{len(df)}")
print(f"有效類別數：{df['label'].nunique()}")

# 特徵和標籤
X = df.drop('label', axis=1).values
y = df['label'].values

# 標籤編碼
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 特徵標準化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 切分訓練/測試集
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
print(f"\n訓練集：{len(X_train)} 筆")
print(f"測試集：{len(X_test)} 筆")

# 訓練 MLP 分類器
print("\n開始訓練 MLP 分類器...")
clf = MLPClassifier(
    hidden_layer_sizes=(256, 128, 64),
    activation='relu',
    max_iter=200,
    random_state=42,
    verbose=True,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=10
)

clf.fit(X_train, y_train)

# 評估
print("\n評估模型...")
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"測試集準確率：{acc:.4f} ({acc*100:.2f}%)")

# 儲存模型
os.makedirs("models", exist_ok=True)
with open("models/classifier.pkl", "wb") as f:
    pickle.dump(clf, f)
with open("models/label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)
with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("\n模型已儲存到 models/")
print(f"  classifier.pkl")
print(f"  label_encoder.pkl")
print(f"  scaler.pkl")