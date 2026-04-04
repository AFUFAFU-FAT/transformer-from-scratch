"""
診斷：找 的手指二值 / 再見 喜歡 的信心問題
"""
import numpy as np
import pandas as pd

FINGER_JOINTS = [(2,3,4),(5,6,7),(9,10,11),(13,14,15),(17,18,19)]
TIP_IDX = [4, 8, 12, 16, 20]
MCP_IDX = [1, 5,  9, 13, 17]
LABELS = ["拇","食","中","無","小"]

def pip_angles(lm63):
    """回傳各手指 PIP 關節角度（度數）"""
    lms = lm63.reshape(21, 3)
    angles = np.empty(5, dtype=np.float32)
    for fi, (a, b, c) in enumerate(FINGER_JOINTS):
        v1 = lms[a] - lms[b]; v2 = lms[c] - lms[b]
        cos_a = np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2)+1e-6)
        angles[fi] = np.degrees(np.arccos(np.clip(cos_a,-1.0,1.0)))
    return angles

def extension_scores(lm63):
    """tip-to-wrist / MCP-to-wrist 比值（>1 代表伸出）"""
    lms = lm63.reshape(21, 3); wrist = lms[0]
    scores = np.empty(5, dtype=np.float32)
    for fi, (ti, mi) in enumerate(zip(TIP_IDX, MCP_IDX)):
        scores[fi] = np.linalg.norm(lms[ti]-wrist) / (np.linalg.norm(lms[mi]-wrist)+1e-6)
    return scores

def binary_at_threshold(lm63, thresh):
    angles = pip_angles(lm63)
    return (angles > thresh).astype(np.float32)

df = pd.read_csv("data/recorded_features.csv")

# ──────────────────────────────────────────────────────────────
# 1. 找：PIP 角度分析（判斷 EXTEND_ANGLE 是否合適）
# ──────────────────────────────────────────────────────────────
print("=" * 60)
print("  找  的手指 PIP 角度分析")
print("=" * 60)
rows = df[df["label"] == "找"]
if len(rows) == 0:
    print("  [找] 無資料")
else:
    r_coords = rows[[f"lm_{i}" for i in range(63)]].values
    all_angles = np.array([pip_angles(r) for r in r_coords])  # (N, 5)
    all_ext    = np.array([extension_scores(r) for r in r_coords])

    print(f"  樣本數：{len(rows)} 幀")
    print(f"\n  PIP 角度（度數）—— 期望 中/無/小 接近 180°，拇/食 < 140°")
    for fi in range(5):
        ang = all_angles[:, fi]
        frac_140 = (ang > 140).mean()
        frac_130 = (ang > 130).mean()
        frac_120 = (ang > 120).mean()
        print(f"    {LABELS[fi]}指: avg={ang.mean():.1f}°  std={ang.std():.1f}°  "
              f">140°:{frac_140*100:.0f}%  >130°:{frac_130*100:.0f}%  >120°:{frac_120*100:.0f}%")

    print(f"\n  Extension scores（tip/MCP，>1=伸出）")
    for fi in range(5):
        sc = all_ext[:, fi]
        print(f"    {LABELS[fi]}指: avg={sc.mean():.2f}  std={sc.std():.2f}  >1.2:{(sc>1.2).mean()*100:.0f}%")

    print(f"\n  在不同閾值下 binary 的平均（期望 中無小 → 1, 拇食 → 0）")
    for thresh in [120, 130, 140, 150]:
        bins = np.array([binary_at_threshold(r, thresh) for r in r_coords]).mean(0)
        print(f"    閾值{thresh}°: [{' '.join(f'{v:.2f}' for v in bins)}]  "
              f"(應為 0 0 1 1 1)")

# ──────────────────────────────────────────────────────────────
# 2. 再見：特徵分析（為何信心低？）
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  再見  信心低原因分析")
print("=" * 60)
rows_zj = df[df["label"] == "再見"]
if len(rows_zj) == 0:
    print("  [再見] 無資料")
else:
    r_coords = rows_zj[[f"lm_{i}" for i in range(63)]].values
    all_angles = np.array([pip_angles(r) for r in r_coords])
    all_ext    = np.array([extension_scores(r) for r in r_coords])

    # 手腕位置變化（搖手 → 高 motion variance）
    r_wrist = rows_zj[["lm_0","lm_1","lm_2"]].values
    dx = np.diff(r_wrist[:, 0]); dy = np.diff(r_wrist[:, 1]); dz = np.diff(r_wrist[:, 2])
    motion = np.sqrt(dx**2 + dy**2 + dz**2)

    # 食指指向 z
    ptr_z = rows_zj["lm_65"].values if "lm_65" in rows_zj.columns else None

    print(f"  樣本數：{len(rows_zj)} 幀")
    print(f"\n  PIP 角度（期望全部手指伸直 → 全部接近 180°）")
    for fi in range(5):
        ang = all_angles[:, fi]
        frac = (ang > 140).mean()
        print(f"    {LABELS[fi]}指: avg={ang.mean():.1f}°  >140°:{frac*100:.0f}%")

    print(f"\n  Extension scores")
    for fi in range(5):
        sc = all_ext[:, fi]
        print(f"    {LABELS[fi]}指: avg={sc.mean():.2f}  >1.2:{(sc>1.2).mean()*100:.0f}%")

    print(f"\n  Motion（搖手 → 幀間移動量大、方差大）")
    print(f"    avg={motion.mean():.4f}  std={motion.std():.4f}  max={motion.max():.4f}")
    print(f"    高 motion（>0.02）的幀比例：{(motion>0.02).mean()*100:.0f}%")

    # 和哪些詞最相似？（比較 binary mean）
    print(f"\n  與其他詞的 binary 差異（找出最容易混淆的）")
    zj_bin_mean = np.array([binary_at_threshold(r, 140) for r in r_coords]).mean(0)
    print(f"    再見 binary 均值: [{' '.join(f'{v:.2f}' for v in zj_bin_mean)}]")
    for compare_word in ["他","你","我","是","說","問","找","好"]:
        c_rows = df[df["label"] == compare_word]
        if len(c_rows) == 0: continue
        c_coords = c_rows[[f"lm_{i}" for i in range(63)]].values
        c_bin = np.array([binary_at_threshold(r, 140) for r in c_coords]).mean(0)
        diff = np.abs(zj_bin_mean - c_bin).sum()
        print(f"    vs {compare_word}: binary diff={diff:.2f}  [{' '.join(f'{v:.2f}' for v in c_bin)}]")

# ──────────────────────────────────────────────────────────────
# 3. 喜歡：特徵分析（為何信心低？）
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  喜歡  信心低原因分析")
print("=" * 60)
rows_xh = df[df["label"] == "喜歡"]
if len(rows_xh) == 0:
    print("  [喜歡] 無資料")
else:
    r_coords_xh = rows_xh[[f"lm_{i}" for i in range(63)]].values
    l_coords_xh = rows_xh[[f"lm_{i}" for i in range(68, 131)]].values
    r_wrist_xh  = rows_xh[["lm_0","lm_1","lm_2"]].values

    dx = np.diff(r_wrist_xh[:, 0]); dy = np.diff(r_wrist_xh[:, 1])
    motion_xh = np.sqrt(dx**2 + dy**2)
    left_absent = (np.abs(l_coords_xh).sum(axis=1) < 1e-6).mean()

    all_angles_xh = np.array([pip_angles(r) for r in r_coords_xh])
    all_ext_xh    = np.array([extension_scores(r) for r in r_coords_xh])
    pinch_xh = np.array([
        np.linalg.norm(r.reshape(21,3)[4] - r.reshape(21,3)[8]) /
        (np.linalg.norm(r.reshape(21,3)[9] - r.reshape(21,3)[0]) + 1e-6)
        for r in r_coords_xh
    ])

    print(f"  樣本數：{len(rows_xh)} 幀  左手缺失率：{left_absent*100:.0f}%")
    print(f"\n  PIP 角度（喜歡 = 拇食 捏合，其他彎曲）")
    for fi in range(5):
        ang = all_angles_xh[:, fi]
        print(f"    {LABELS[fi]}指: avg={ang.mean():.1f}°  >140°:{(ang>140).mean()*100:.0f}%")
    print(f"\n  Extension scores")
    for fi in range(5):
        sc = all_ext_xh[:, fi]
        print(f"    {LABELS[fi]}指: avg={sc.mean():.2f}  >1.2:{(sc>1.2).mean()*100:.0f}%")
    print(f"\n  Pinch 距離：avg={pinch_xh.mean():.2f}  std={pinch_xh.std():.2f}")
    print(f"  Motion (XY): avg={motion_xh.mean():.4f}  std={motion_xh.std():.4f}")

    # 和「好」比較
    rows_hao = df[df["label"] == "好"]
    if len(rows_hao) > 0:
        hao_coords = rows_hao[[f"lm_{i}" for i in range(63)]].values
        hao_angles = np.array([pip_angles(r) for r in hao_coords])
        hao_ext    = np.array([extension_scores(r) for r in hao_coords])
        hao_pinch  = np.array([
            np.linalg.norm(r.reshape(21,3)[4] - r.reshape(21,3)[8]) /
            (np.linalg.norm(r.reshape(21,3)[9] - r.reshape(21,3)[0]) + 1e-6)
            for r in hao_coords
        ])
        hao_wrist  = rows_hao[["lm_0","lm_1","lm_2"]].values
        hdx = np.diff(hao_wrist[:,0]); hdy = np.diff(hao_wrist[:,1])
        motion_hao = np.sqrt(hdx**2 + hdy**2)
        print(f"\n  【好】vs 【喜歡】對比：")
        print(f"    好   Motion: avg={motion_hao.mean():.4f}  好 Pinch: avg={hao_pinch.mean():.2f}")
        print(f"    喜歡 Motion: avg={motion_xh.mean():.4f}  喜歡 Pinch: avg={pinch_xh.mean():.2f}")
        print(f"    好   binary: [{' '.join(f'{v:.2f}' for v in np.array([binary_at_threshold(r,140) for r in hao_coords]).mean(0))}]")
        print(f"    喜歡 binary: [{' '.join(f'{v:.2f}' for v in np.array([binary_at_threshold(r,140) for r in r_coords_xh]).mean(0))}]")

# ──────────────────────────────────────────────────────────────
# 4. 樣本數統計
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  各詞彙樣本數（排序）")
print("=" * 60)
counts = df["label"].value_counts()
for word in ["找","再見","喜歡","好","他","說","是","對不起"]:
    n = counts.get(word, 0)
    print(f"  {word}: {n} 幀")
print(f"\n  全部詞彙平均：{counts.mean():.0f} 幀  中位數：{counts.median():.0f} 幀")
