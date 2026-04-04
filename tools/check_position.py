"""
檢查 是/再見/喜歡/不喜歡 的手腕位置和手臉距離特徵
dims 161:167 = 右手腕到[鼻/嘴/左耳/右耳/左眼/右眼] 距離
dims 131:134 = 右手腕 rel_wrist (x,y,z) 相對身體
"""
import numpy as np
import pandas as pd

df = pd.read_csv("data/recorded_features.csv")

# face_cols: lm_63~66 是 pose 的 nose/mouth/ear landmarks (在 recorded_features 裡)
# 不過 recorded_features 只存原始座標，要看 build_sequences 輸出的特徵
# 改用原始 landmark 算手臉距離

# pose landmarks: lm_63(nose_x), lm_64(nose_y), lm_65(nose_z), lm_66~67(pose meta)
# right wrist = lm_0,1,2; nose = lm_63,64,65
face_landmarks = {
    "nose":  (63, 64, 65),
}

for word in ["喜歡", "不喜歡", "是", "再見"]:
    rows = df[df["label"] == word]
    if len(rows) == 0:
        print(f"{word}: 無資料"); continue

    r_wrist = rows[["lm_0","lm_1","lm_2"]].values        # 右手腕
    nose    = rows[["lm_63","lm_64","lm_65"]].values      # 鼻子

    dist_nose = np.linalg.norm(r_wrist - nose, axis=1)

    # 右手腕 y 座標（越小越高，因為 y 軸朝下）
    wrist_y = rows["lm_1"].values

    print(f"\n【{word}】{len(rows)} 幀")
    print(f"  手腕到鼻距離:  avg={dist_nose.mean():.3f}  std={dist_nose.std():.3f}")
    print(f"  手腕 y 座標:   avg={wrist_y.mean():.3f}  std={wrist_y.std():.3f}  (越小=越高)")
