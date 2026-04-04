"""
檢查訓練 CSV 中指定詞彙的 binary 特徵分佈
確認 build_sequences 是否正確計算 finger binary
"""
import numpy as np
import pandas as pd

FINGER_JOINTS = [(2,3,4),(5,6,7),(9,10,11),(13,14,15),(17,18,19)]
EXTEND_ANGLE  = 140.0

def compute_finger_binary(lm63):
    lms = lm63.reshape(21, 3)
    binary = np.empty(5, dtype=np.float32)
    for fi, (a, b, c) in enumerate(FINGER_JOINTS):
        v1 = lms[a] - lms[b]; v2 = lms[c] - lms[b]
        cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        binary[fi] = 1.0 if np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))) > EXTEND_ANGLE else 0.0
    return binary

def compute_pinch(lm63):
    lms = lm63.reshape(21, 3)
    dist = np.linalg.norm(lms[4] - lms[8]) / (np.linalg.norm(lms[9] - lms[0]) + 1e-6)
    return min(dist, 3.0)

df = pd.read_csv("data/recorded_features.csv")

for word in ["好", "喜歡", "謝謝", "認真", "不是", "對不起", "學", "再見", "甚麼", "現在", "晚上"]:
    rows = df[df["label"] == word]
    if len(rows) == 0:
        print(f"{word}: 無資料"); continue

    # 右手（lm_0~62）
    r_coords = rows[[f"lm_{i}" for i in range(63)]].values
    r_bin  = np.array([compute_finger_binary(r) for r in r_coords])
    pinches = np.array([compute_pinch(r) for r in r_coords])
    r_mean = r_bin.mean(axis=0)

    # 左手（lm_68~130）
    l_coords = rows[[f"lm_{i}" for i in range(68, 131)]].values
    l_bin  = np.array([compute_finger_binary(r) for r in l_coords])
    l_mean = l_bin.mean(axis=0)

    # 左手全零（未偵測到）的幀比例
    left_absent = (np.abs(l_coords).sum(axis=1) < 1e-6).mean()

    labels = "拇食中無小"
    print(f"\n【{word}】{len(rows)} 幀")
    print(f"  右手 binary: [{' '.join(f'{v:.2f}' for v in r_mean)}]")
    print(f"  左手 binary: [{' '.join(f'{v:.2f}' for v in l_mean)}]  ← 左手未偵測率: {left_absent*100:.0f}%")
    print(f"  pinch 距離:  平均={pinches.mean():.2f}  最小={pinches.min():.2f}  最大={pinches.max():.2f}")
