import numpy as np, pickle

data = np.load('data/sequences.npz', allow_pickle=True)
with open('data/seq_label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

print("npz keys:", list(data.keys()))
labels = np.concatenate([data['y_train'], data['y_test']])
print(f"總樣本數: {len(labels)}\n")
counts = [(cls, int((labels == i).sum())) for i, cls in enumerate(le.classes_)]
counts.sort(key=lambda x: x[1])
for cls, n in counts:
    bar = '█' * (n // 2)
    print(f"{cls:8s} {n:4d}  {bar}")
