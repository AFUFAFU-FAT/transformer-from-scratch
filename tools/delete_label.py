import pandas as pd, sys

label = sys.argv[1] if len(sys.argv) > 1 else "找"
df = pd.read_csv("data/recorded_features.csv")
before = len(df)
df = df[df["label"] != label]
df.to_csv("data/recorded_features.csv", index=False)
print(f"刪除「{label}」：{before} → {len(df)} 筆")
