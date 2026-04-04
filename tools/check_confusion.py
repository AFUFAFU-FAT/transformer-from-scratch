import numpy as np, pickle, torch, torch.nn as nn

class SignLanguageBiLSTM(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=64, num_layers=1, dropout=0.6):
        super().__init__()
        self.input_proj = nn.Sequential(nn.Linear(input_dim, 64), nn.LayerNorm(64), nn.ReLU())
        self.lstm = nn.LSTM(input_size=64, hidden_size=hidden_dim, num_layers=num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0, bidirectional=True)
        self.attention  = nn.Linear(hidden_dim * 2, 1)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim * 2, num_classes))
        self.dropout    = nn.Dropout(dropout)
    def forward(self, x):
        x = self.input_proj(x)
        out, _ = self.lstm(x)
        out = self.dropout(out)
        attn = torch.softmax(self.attention(out), dim=1)
        return self.classifier((attn * out).sum(dim=1))

with open('models/lstm_config.pkl','rb') as f: cfg = pickle.load(f)
with open('data/seq_label_encoder.pkl','rb') as f: le  = pickle.load(f)
data = np.load('data/sequences.npz', allow_pickle=True)

device = torch.device('cpu')
model = SignLanguageBiLSTM(cfg['input_dim'], cfg['num_classes'],
                           cfg['hidden_dim'], cfg['num_layers'], cfg['dropout'])
ckpt = torch.load('models/lstm_best.pth', map_location=device, weights_only=False)
model.load_state_dict(ckpt['model_state_dict']); model.eval()

feat_mean = torch.FloatTensor(cfg['feat_mean'])
feat_std  = torch.FloatTensor(cfg['feat_std'])
SEQ_LEN   = cfg['seq_len']

X, y = data['X_test'], data['y_test']

# add delta
delta = np.zeros_like(X)
delta[:, 1:, :] = X[:, 1:, :] - X[:, :-1, :]
X = np.concatenate([X, delta], axis=2)

# add cumulative displacement (right wrist 63:66, left wrist 131:134)
right_start = X[:, 0:1, 63:66]
left_start  = X[:, 0:1, 131:134]
cum_r = X[:, :, 63:66]   - right_start
cum_l = X[:, :, 131:134] - left_start
X = np.concatenate([X, cum_r, cum_l], axis=2)

wrong = {}
with torch.no_grad():
    for i in range(len(X)):
        x = torch.FloatTensor(X[i]).unsqueeze(0)
        x = (x - feat_mean) / feat_std
        pred = model(x).argmax(1).item()
        true = int(y[i])
        if pred != true:
            key = (le.classes_[true], le.classes_[pred])
            wrong[key] = wrong.get(key, 0) + 1

print("混淆對（真實 → 預測）：")
for (t, p), n in sorted(wrong.items(), key=lambda x: -x[1])[:20]:
    print(f"  {t:8s} → {p:8s}  {n}次")
