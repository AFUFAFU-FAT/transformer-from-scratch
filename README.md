---
title: Taiwan Sign Language Recognition
emoji: 🤟
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# 台灣手語即時辨識 AI 系統

即時辨識台灣手語詞彙，透過瀏覽器攝影機擷取畫面，以 MediaPipe 提取手部與身體特徵，送入 BiLSTM 模型分類，結果即時顯示於網頁。

[點我前往 台灣手語辨識網頁](https://huggingface.co/spaces/AFUFAFU/TWSL_AI_Project)

> 自主學習計畫作品：從閱讀 *Attention Is All You Need* 論文、手推 Transformer 公式，到將 Attention 機制應用於手語序列辨識模型，實作完整的「資料收集 → 特徵工程 → 模型訓練 → 即時部署」AI 開發流程。

[![Built with Claude Code](https://img.shields.io/badge/Built%20with-Claude%20Code-black?logo=anthropic)](https://claude.ai/code)

---

## 功能

- 瀏覽器攝影機即時辨識，無需安裝任何軟體
- 支援 **32 個台灣手語常用詞彙**
- Top-3 候選詞彙即時顯示，附信心度進度條
- 手停止時自動觸發 EndPose Hybrid 辨識（提升靜態手勢準確率）
- 多層 Gate 系統：手形 Gate → 動作 Gate → Post-hoc 強制替換
- 完整句子累積記錄（無長度限制）
- 頁面內建詞彙字典

---

## 系統架構

```
瀏覽器攝影機（每 100ms 截圖）
    ↓  WebSocket
Flask 後端（app.py）
    ↓
MediaPipe HandLandmarker + PoseLandmarker
→ 199 維特徵向量（雙手關鍵點 + 身體錨點 + 手臂 + 手臉距離
                  + 身體朝向 + 手指伸展分數 + binary + pinch + pointing）
    ↓  + 199 維 delta（幀間差分）+ 6 維 cumulative（累積位移）= 404 維
狀態機（IDLE → COLLECTING → 觸發）
    ↓
BiLSTM + Attention Pooling
→ 多層 Gate（手形/動作/雙手/說/找 專項規則）
→ 是 專項偵測（最高優先）
→ 32 類詞彙輸出
    ↓  WebSocket
瀏覽器即時顯示
```

---

## 從 Transformer 理論到實際應用

本專案的核心模型使用 **BiLSTM + Attention Pooling**。Attention Pooling 直接源自 Transformer 的 Scaled Dot-Product Attention 機制：對序列中每個時間步計算重要性權重，加權求和得到整段手勢的代表向量，再送入分類器。

Day 1 從零推導 Transformer 的目的，正是為了理解這個機制的數學本質——為什麼 Attention 能讓模型自動聚焦在手勢動作最關鍵的幾幀，而不是平等對待所有幀。

```
Transformer Attention（論文）          BiLSTM Attention Pooling（本專案）
───────────────────────────────        ──────────────────────────────────
Q, K, V 計算相似度權重                  對每個時間步的 hidden state 計算權重
softmax 正規化                          softmax 正規化
加權求和 → context vector              加權求和 → 手勢代表向量 → 分類器
```

---

## 開發過程

### Day 1｜理論基礎

閱讀 *Attention Is All You Need*，手推 Scaled Dot-Product Attention、Multi-Head Attention、Positional Encoding、Feed-Forward Network 等所有核心公式，以 NumPy 從零實作 Transformer Encoder，與 PyTorch 官方實作數值比對（最大誤差 4.44e-16）。

### Day 2｜第一次嘗試：網路爬蟲 × 失敗的資料來源

最初的構想是直接使用台灣教育部手語辭典 3,495 個詞彙的示範影片作為訓練資料。

**實作過程：**
- 分析手語辭典網站 API 結構（SPA，以 Playwright 攔截請求）
- 以 requests 直接呼叫 API，下載全部詞彙影片與語言學結構化資料
- MediaPipe VIDEO 串流模式從 3,495 部影片提取 160,157 筆關鍵點特徵
- 訓練 MLP → BiLSTM，觀察指標

**根本問題：Domain Gap**

辭典影片由專業手語老師在攝影棚、固定角度、標準姿勢拍攝，與自己坐在桌前對著筆電攝影機比手語的動作分布完全不同。模型在辭典影片訓練得再好，即時辨識信心度只有 6–15%，無法實用。外部公開資料集的分布與真實使用場景差距太大，唯一解法是自己錄製訓練資料。

### Day 3｜重新出發：自錄資料 × 特徵工程 × 部署

放棄爬蟲方案後，重新設計整個 pipeline。

**資料收集：**

以「動作誇張、讓手勢間差異明顯」的方式自錄 32 個詞彙，每詞錄製 15 次，並在每次刻意改變位置（偏左偏右偏高偏低）、速度（快打慢打）和距離（靠近/遠離鏡頭），讓模型學到手勢本質特徵而非特定錄影條件下的規律。

**解決訓練流程 bug：**

多次重錄後混淆次數不降反升，最終發現 `build_sequences.py` 讀取的是 `features_filtered.csv`，但流程中漏掉了 `filter_vocab.py` 這一步，所有重錄資料從未進入訓練。修正後完整流程：

```
record_vocab.py → filter_vocab.py → build_sequences.py → train_lstm.py
```

**解決過擬合：**

初版以 STRIDE=4 的滑動窗口切分序列，相鄰序列有 87.5% 的幀重疊，模型直接記住答案。改為 STRIDE=16 後過擬合消失，驗證準確率穩定在 **95.31%**。

**EndPose Hybrid（即時辨識優化）：**

偵測到手腕靜止時，同時以兩種方式推論並加權合併：

```
最終分數 = 完整序列推論 × 0.6 + 靜止幀重複推論 × 0.4
```

若結尾姿勢信心度低（< 0.30，表示手勢結尾不固定如搖手），自動調整為 75/25，以序列主導避免不穩定 endpose 拉低整體信心。

---

## 特徵工程（199 維 → 訓練時 404 維）

| 區塊 | 維度 | 說明 |
|------|------|------|
| 右手關鍵點 | 0–62 | 21 landmarks × 3，以手腕為原點/掌長正規化 |
| 右手 rel_wrist | 63–65 | 相對肩寬正規化（動作偵測用） |
| 右手 palm_dir | 66–67 | 掌心朝向單位向量 |
| 左手關鍵點 | 68–130 | 同右手 |
| 左手 rel_wrist | 131–133 | |
| 左手 palm_dir | 134–135 | |
| 身體錨點 | 136–145 | 鼻、肩、髖位置 |
| 保留 | 146–148 | 補零 |
| 雙臂方向 | 149–160 | 肘位置 + 上臂/前臂方向向量 |
| 手臉距離 | 161–166 | 手腕→鼻/嘴/耳（÷肩寬正規化） |
| 身體朝向 | 167–168 | 肩膀向量 cos/sin |
| 手指伸展分數 | 169–178 | tip/mcp 距離比（runtime 推導） |
| 手指 binary | 179–188 | PIP 角度 > 140° → 1（5指 × 雙手） |
| 手指指向方向 | 189–194 | normalize(tip8−MCP5)（雙手） |
| pinch 距離 | 195–196 | 拇食距離/掌長（雙手） |
| is_right/left | 197–198 | 手部偵測旗標 |
| +199 delta | 199–397 | feat[t] − feat[t-1] |
| +6 cumulative | 398–403 | 右/左手腕相對起點累積位移 |

### 特徵權重設計

```python
W_BINARY    = 10.0  # 手指 binary（最高優先，直接判斷哪根伸出）
W_POINTING  =  8.0  # 食指指向方向（你/我/他/說 的關鍵）
W_HANDSHAPE =  6.0  # 手指伸展分數（量化手形）
W_FINGERTIP =  4.0  # 指尖座標
W_FACE      =  5.0  # 手腕到臉距離（爸爸/媽媽/謝謝 等貼臉手勢）
W_POSITION  =  4.0  # 手腕相對身體位置
W_FINGER    =  3.0  # 其他手部關節
W_ARM       =  2.5  # 手臂方向
W_ORIENT    =  2.0  # 身體朝向
W_ANCHOR    =  1.5  # 身體錨點
```

---

## 多層 Gate 系統

單純靠 BiLSTM 無法區分手形高度相似的詞彙（如「是」vs「再見」vs「說」），因此加入多層規則 Gate：

### apply_feature_gates（軟壓制，prob × 0.05）

| 詞彙 | 條件 |
|------|------|
| 找 | 食指彎（bin < 3）AND 中/無/小指伸出（bin > 7） |
| 認真 | 左手存在（is_left > 5） |
| 甚麼 | 拇指伸出（bin > 3） |
| 他 | 拇指未明顯伸出（bin < 6） |
| 媽媽 | 小指伸出（bin > 3） |
| 再見 | 手腕 X 方向標準差 > 0.06（搖手動作） |
| 喜歡 | 手腕 X 或 Y 標準差 > 0.05（上下/左右動作） |
| 是 | 手腕 X 標準差 < 0.12 AND Y < 0.12（靜止，排除再見） |
| 好 | 手腕 X 標準差 < 0.10 AND Y < 0.10（靜止，排除喜歡） |

### apply_gates_posthoc（強制替換）

- **Gate 1（雙手）**：左手出現率 < 40% → 強制排除「對不起/認真/人/聽人/晚上」
- **Gate 3（說）**：≥ 4 根手指伸出 → 換成下一個候選
- **Gate 4（找）**：食指彎 + 中無小指伸出 + 無左手 → 強制輸出「找」

### 是 專項偵測（最高優先）

「是」與「說」的 binary 幾乎相同，模型信心持續偏低。專項偵測在所有其他 Gate 之後執行：

- **來源一**：candidates top3 中「是」≥ 8% → 強制輸出
- **來源二**：直查模型機率向量「是」≥ 5% → 強制輸出
- 任一成立即覆蓋其他所有選項

---

## 模型規格

| 項目 | 說明 |
|------|------|
| 架構 | BiLSTM（hidden=64, layers=1）+ Attention Pooling |
| 輸入維度 | 404（199 base + 199 delta + 6 cumulative） |
| 詞彙數 | 32 個台灣手語詞彙 |
| 驗證準確率 | 95.31% |
| 序列長度 | 32 幀 |
| 訓練資料 | 32 詞彙 × 15 次自錄，多位置/速度/距離 |

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

## 安裝與執行

### 環境需求

- Python 3.10+
- CUDA（選用，CPU 亦可執行）

### 安裝

```bash
pip install -r requirements.txt
```

### 執行

```bash
# Web 伺服器（瀏覽器辨識）
python app.py
# 瀏覽器開啟 http://localhost:5000

# 本機即時辨識（不需瀏覽器）
python recognize_endpose.py

# 公開部署（ngrok）
ngrok http 5000
```

### 重新訓練

```bash
python record_vocab.py       # 錄製訓練資料
python filter_vocab.py       # 篩選詞彙（不可略過）
python build_sequences.py    # 建立序列
python train_lstm.py         # 訓練模型
```

---

## 專案結構

```
├── app.py                   Flask + SocketIO 後端
├── templates/
│   └── index.html           前端 Web UI
├── recognize_endpose.py     本機即時辨識（含 Gate 系統 + EndPose Hybrid）
├── recognize.py             ⚠️ 舊版辨識器（保留供參考）
├── lm_selector.py           LLM 評分模組（目前停用）
├── record_vocab.py          錄製訓練資料
├── filter_vocab.py          詞彙篩選
├── build_sequences.py       序列建立與資料增強
├── train_lstm.py            BiLSTM 訓練
├── mini_transformer.py      Day 1：從零實作 Transformer
├── notion_update.py         Notion 開發紀錄更新工具
├── models/
│   ├── lstm_best.pth        訓練好的 BiLSTM 權重
│   └── lstm_config.pkl      模型設定與正規化參數
├── data/
│   └── seq_label_encoder.pkl  詞彙標籤對照
├── tools/                   診斷工具（不影響主程式）
│   ├── check_binary.py      手指 binary 分佈分析
│   ├── check_confusion.py   混淆矩陣
│   ├── check_feat.py        混淆詞對深入比較
│   ├── check_samples.py     各詞彙樣本數統計
│   ├── check_zhao_conf.py   找/再見/喜歡 PIP 角度診斷
│   ├── check_position.py    手腕位置分佈分析
│   ├── count_label.py       指定詞彙幀數確認
│   └── delete_label.py      刪除指定詞彙資料
├── _unused/                 開發過程被捨棄的分支（爬蟲、CNN、Qwen 等）
└── Dockerfile               Hugging Face Spaces 部署
```

---

## 已知限制

- app.py 為單一 global state，不支援多人同時連線
- 拇指 IP 角度法對內收/外展無效（dim 179 幾乎恆=1），gate 不依賴拇指 binary
- 部分低頻詞彙偶爾信心度偏低，可透過增加錄製次數改善
- LLM 評分（lm_selector.py）目前停用（`LLM_ENABLED=False`）

---

## 核心工程挑戰與解法

| 問題 | 根本原因 | 解法 |
|------|---------|------|
| 即時辨識信心度 6–15% | Domain Gap（訓練/使用環境分布不同） | 改用自錄資料 |
| 混淆次數不降反升 | filter_vocab.py 被跳過，重錄資料未進入訓練 | 完整流程：record → **filter** → build → train |
| 找 被誤判為 再見 | 舉手過渡幀的 binary 和再見相同 | RAMP_FRAMES=6，丟棄序列開頭幀 |
| 是 被誤判為 說 | binary 幾乎相同，模型信心低 | 專項偵測：candidates ≥8% 或直查 ≥5% → 強制輸出 |
| 過擬合（87.5% 幀重疊） | STRIDE=4 滑動窗口 | 改為 STRIDE=16 |

---

[![Built with Claude Code](https://img.shields.io/badge/Built%20with-Claude%20Code-black?logo=anthropic)](https://claude.ai/code)
