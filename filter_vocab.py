"""
filter_vocab.py
從 recorded_features.csv 篩選 30 個目標詞彙
輸出 data/features_filtered.csv，供 build_sequences.py 使用

若要新增/移除詞彙，同步修改 record_vocab.py 的 VOCAB 和這裡的 KEEP_VOCAB
"""
import pandas as pd

# 與 record_vocab.py 的 ALL_VOCAB 同步（30 個）
KEEP_VOCAB = [
    # U1
    "你", "我", "他", "我們",
    "好", "謝謝", "對不起", "再見",
    "喜歡", "不喜歡", "想", "是", "不是", "學", "找",
    "開心", "認真",
    "一樣", "嗎",
    # "聾人",  # 暫停用：與「是」手勢特徵太相似，持續混淆
    "聽人",
    "現在", "晚上",
    # U2
    "介紹", "問", "說", "住",
    "甚麼",
    "地方", "家",
    # U3
    "人", "爸爸", "媽媽",
]

df = pd.read_csv('data/recorded_features.csv')
df_filtered = df[df['label'].isin(KEEP_VOCAB)]

print(f"原始詞彙數：{df['label'].nunique()}  原始筆數：{len(df):,}")
print(f"保留詞彙數：{df_filtered['label'].nunique()}  篩選後筆數：{len(df_filtered):,}")

missing = set(KEEP_VOCAB) - set(df_filtered['label'].unique())
if missing:
    print(f"⚠️  以下詞彙在 CSV 中找不到資料：{missing}")

print(f"\n詞彙分佈：")
print(df_filtered['label'].value_counts().to_string())

df_filtered.to_csv('data/features_filtered.csv', index=False)
print(f"\n✅ 儲存至 data/features_filtered.csv")