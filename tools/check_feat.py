"""
深入檢查混淆詞對的 binary / 位置 / delta(動作) 特徵
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

PAIRS = [
    ("他", "甚麼"),
    ("好", "聽人"),
    ("問", "是"),
    ("說", "人"),
]

for w1, w2 in PAIRS:
    print(f"\n{'='*60}")
    print(f"  {w1}  vs  {w2}")
    print(f"{'='*60}")

    for word in [w1, w2]:
        rows = df[df["label"] == word]
        if len(rows) == 0:
            print(f"  [{word}]: 無資料"); continue

        r_coords = rows[[f"lm_{i}" for i in range(63)]].values
        l_coords = rows[[f"lm_{i}" for i in range(68, 131)]].values

        r_bin   = np.array([compute_finger_binary(r) for r in r_coords])
        pinches = np.array([compute_pinch(r) for r in r_coords])

        left_absent = (np.abs(l_coords).sum(axis=1) < 1e-6).mean()

        r_wrist   = rows[["lm_0","lm_1","lm_2"]].values
        nose      = rows[["lm_63","lm_64","lm_65"]].values
        dist_nose = np.linalg.norm(r_wrist - nose, axis=1)
        wrist_y   = rows["lm_1"].values

        dx = np.diff(r_wrist[:, 0])
        dy = np.diff(r_wrist[:, 1])
        dz = np.diff(r_wrist[:, 2])
        motion_mag = np.sqrt(dx**2 + dy**2 + dz**2)

        print(f"\n  [{word}]  {len(rows)} frames  left_absent={left_absent*100:.0f}%")
        print(f"    binary:   [{' '.join(f'{v:.2f}' for v in r_bin.mean(0))}]")
        print(f"    pinch:    avg={pinches.mean():.2f}  std={pinches.std():.2f}")
        print(f"    nose_d:   avg={dist_nose.mean():.3f}  std={dist_nose.std():.3f}")
        print(f"    wrist_y:  avg={wrist_y.mean():.3f}  std={wrist_y.std():.3f}  (bigger=lower)")
        print(f"    motion:   avg={motion_mag.mean():.4f}  std={motion_mag.std():.4f}  max={motion_mag.max():.4f}")
