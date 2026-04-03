# 台灣手語即時辨識 AI 系統

即時辨識台灣手語詞彙，透過瀏覽器攝影機擷取畫面，以 MediaPipe 提取手部與身體特徵，送入 BiLSTM 模型分類，結果即時顯示於網頁。

> 自主學習計畫作品：從閱讀 *Attention Is All You Need* 論文、手推 Transformer 公式，到將 Attention 機制應用於手語序列辨識模型，3 天內完成可公開使用的 AI 網站。

---

## 功能

- 瀏覽器攝影機即時辨識，無需安裝任何軟體
- 支援 32 個台灣手語常用詞彙
- Top-3 候選詞彙即時顯示，附信心度進度條
- 手停止時自動觸發 EndPose Hybrid 辨識（提升靜態手勢準確率）
- 詞彙累積記錄，形成句子
- 頁面內建詞彙字典，告知使用者可辨識範圍

---

## 系統架構

```
瀏覽器攝影機（每 100ms 截圖）
    ↓  WebSocket
Flask 後端（app.py）
    ↓
MediaPipe HandLandmarker + PoseLandmarker
→ 179 維特徵向量
    ↓
狀態機（IDLE → COLLECTING → 觸發）
    ↓
BiLSTM + Attention Pooling
→ 32 類詞彙 Top-3
    ↓  WebSocket
瀏覽器即時顯示
```

---

## 32 個可辨識詞彙

| 類別 | 詞彙 |
|------|------|
| 人稱 | 你、我、他、我們、人、聽人 |
| 問候／禮貌 | 好、謝謝、對不起、再見、開心、認真 |
| 動詞 | 喜歡、不喜歡、想、是、不是、學、找、介紹、問、說、住 |
| 家庭 | 爸爸、媽媽、家 |
| 時間／地點 | 現在、晚上、地方、一樣 |
| 疑問／其他 | 嗎、甚麼 |

---

## 技術細節

### 特徵工程（179 維）

| 區塊 | 維度 | 說明 |
|------|------|------|
| 雙手關鍵點 | 0–135 | 21 landmarks × 3 + rel_wrist + palm_dir，各手 68 維 |
| 身體錨點 | 136–145 | 鼻、肩、髖位置 |
| 雙臂方向 | 149–160 | 肘位置 + 上臂 / 前臂方向向量 |
| 手臉距離 | 161–166 | 手腕到鼻 / 嘴 / 耳的距離（區分貼臉手勢） |
| 身體朝向 | 167–168 | 肩膀向量 cos/sin（runtime 推導） |
| 手指伸展分數 | 169–178 | tip/mcp 距離比，量化每指伸展程度（runtime 推導） |

### 模型

- **架構**：BiLSTM（hidden=64）+ Attention Pooling
- **訓練資料**：32 詞彙 × 15 次自錄，多位置、多速度
- **驗證準確率**：95.31%
- **序列長度**：32 幀（STRIDE=16 滑動窗口）

### EndPose Hybrid

手腕靜止超過 5 幀時，同時用「完整序列」和「靜止幀重複」兩種方式推論，加權合併（60% / 40%），讓靜態手勢信心度更穩定。

---

## 從 Transformer 理論到實際應用

本專案的核心模型使用 **BiLSTM + Attention Pooling**。Attention Pooling 直接源自 Transformer 的 Scaled Dot-Product Attention：對序列中每個時間步計算重要性權重，加權求和得到整段手勢的代表向量，再送入分類器。

Day 1 從零推導 Transformer 的目的，正是為了理解這個機制的數學本質——為什麼 Attention 能讓模型自動聚焦在手勢動作最關鍵的幾幀，而不是平均對待所有幀。

```
Transformer Attention（論文）          BiLSTM Attention Pooling（本專案）
───────────────────────────────        ──────────────────────────────────
Q, K, V 計算相似度權重                  對每個時間步的 hidden state 計算權重
softmax 正規化                          softmax 正規化
加權求和 → context vector              加權求和 → 手勢代表向量 → 分類
```

---

## 開發過程

整個專案在 **3 天**內完成：

- **Day 1**：閱讀 *Attention Is All You Need*，手推 Scaled Dot-Product Attention、Multi-Head Attention、Positional Encoding 等所有核心公式，以 NumPy 從零實作並與 PyTorch 數值比對（最大誤差 4.44e-16）
- **Day 2–3**：在 [Claude Code](https://claude.ai/code) 輔助下，將 Attention 機制應用於手語序列辨識，完成資料收集、特徵工程、模型訓練到 Web 部署的完整流程

### 主要任務

| 任務 | 說明 |
|------|------|
| 資料收集 | 設計 179 維特徵、自錄 32 詞彙訓練資料（解決 Domain Gap） |
| 特徵工程 | 滑動窗口切分、資料增強、特徵加權設計 |
| 模型訓練 | BiLSTM + Attention Pooling，驗證準確率 95.31% |
| 即時辨識 | 狀態機 + EndPose Hybrid，解決即時信心度不穩問題 |
| Web 部署 | Flask + SocketIO 後端、瀏覽器前端、ngrok 公開連結 |

---

## 安裝與執行

### 環境需求

- Python 3.10+
- CUDA（選用，CPU 亦可執行）

### 安裝

```bash
pip install -r requirements.txt
```

### 執行後端

```bash
python app.py
# 瀏覽器開啟 http://localhost:5000
```

### 公開部署（ngrok）

```bash
ngrok http 5000
```

---

## 專案結構

```
├── app.py                  Flask + SocketIO 後端
├── templates/
│   └── index.html          前端 Web UI
├── record_vocab.py         錄製訓練資料
├── filter_vocab.py         詞彙篩選
├── build_sequences.py      序列建立與資料增強
├── train_lstm.py           BiLSTM 訓練
├── recognize_endpose.py    本機即時辨識（含 EndPose Hybrid）
├── recognize.py            本機即時辨識（基礎版）
├── mini_transformer.py     Day 1：從零實作 Transformer
├── models/                 訓練好的模型權重
├── data/                   訓練資料（CSV / NPZ）
└── _unused/                開發過程中被捨棄的分支（保留供參考）
```

---

## 主要挑戰與解決方式

**Domain Gap**：用 CCU 手語辭典影片訓練的模型，在即時辨識時信心度僅 6–15%。
→ 改為自錄訓練資料，加入身體相對位置正規化。

**過擬合（100% 訓練 / 37% 驗證）**：STRIDE=4 產生 87% 重複序列，模型直接記憶答案。
→ STRIDE 改為 16，縮小模型容量（hidden 256→64）。

**即時信心度不穩**：手勢動作中信心度跳動大。
→ EndPose Hybrid：手停止時合併靜止幀預測，穩定信心度。
