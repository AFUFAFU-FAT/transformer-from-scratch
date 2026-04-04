import pandas as pd, sys

label = sys.argv[1] if len(sys.argv) > 1 else "找"
df = pd.read_csv("data/recorded_features.csv")
print(f"「{label}」目前幀數：{len(df[df['label'] == label])}")
