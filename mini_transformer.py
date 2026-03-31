import numpy as np

# ============================================================
# Mini Transformer — 從零手刻
# ============================================================
# 整體流程：
#
#   輸入 tokens（一排向量，每個代表一個字）
#       ↓
#   加上位置資訊（Positional Encoding）
#       ↓
#   重複 N 層 Encoder Block：
#     ├─ Multi-Head Self-Attention（讓每個字參考其他字）
#     ├─ Add & Norm（Residual + LayerNorm）
#     ├─ Feed-Forward（對每個字做非線性轉換）
#     └─ Add & Norm
#       ↓
#   輸出（每個字都包含了整個句子的上下文資訊）
#
# ============================================================


def softmax(x):
    # 因為 softmax(z_i) = softmax(z_i-c)
    # 所以用 softmax(z_i-max(z)) 的方式使每個元素在e^(負數)~e^0之間
    
    x = x - x.max(axis=-1, keepdims=True) 
    # axis=-1代表僅考慮矩陣最後一行，keepdims代表維持原維度使x-x.max可以成立
    
    e_x = np.exp(x)

    # 每個元素除以該列總和，讓每列加總為 1
    return e_x / e_x.sum(axis=-1, keepdims=True)


def scaled_dot_product_attention(Q, K, V):
    # Q: (seq_len, d_k) — 「我想找什麼」
    # K: (seq_len, d_k) — 「我有什麼」
    # V: (seq_len, d_v) — 「被找到後提供什麼」

    d_k = Q.shape[-1]

    # Step 1: 計算每對 token 的相似度，除以 sqrt(d_k) 防止數值過大
    scores = Q @ K.T / np.sqrt(d_k)       # (seq_len, seq_len)

    # Step 2: 轉成機率分布（每列加總為 1）
    weights = softmax(scores)              # (seq_len, seq_len)

    # Step 3: 用權重對 V 做加權平均
    output = weights @ V                   # (seq_len, d_v)

    return output, weights


def multi_head_attention(x, h, d_model):
    # 把 Attention 拆成 h 個視角平行計算
    # x: (seq_len, d_model)

    d_k = d_model // h   # 每個 head 的維度

    np.random.seed(0)
    # 每個 head 各自的投影矩陣（訓練時這些是被學出來的）
    WQ = [np.random.randn(d_model, d_k) * 0.1 for _ in range(h)]
    WK = [np.random.randn(d_model, d_k) * 0.1 for _ in range(h)]
    WV = [np.random.randn(d_model, d_k) * 0.1 for _ in range(h)]
    WO  = np.random.randn(h * d_k, d_model) * 0.1  # 整合所有 head 的矩陣

    heads = []
    for i in range(h):
        Qi = x @ WQ[i]   # 投影到第 i 個 head 的子空間
        Ki = x @ WK[i]
        Vi = x @ WV[i]
        head_i, _ = scaled_dot_product_attention(Qi, Ki, Vi)
        heads.append(head_i)

    # 把所有 head 的輸出頭尾相接
    concat = np.concatenate(heads, axis=-1)  # (seq_len, h * d_k)

    # 用 WO 把各 head 資訊整合起來
    return concat @ WO                       # (seq_len, d_model)


def positional_encoding(seq_len, d_model):
    # 用 sin/cos 波形告訴模型每個 token 的位置
    # 偶數維用 sin，奇數維用 cos
    # 不同維度對應不同頻率：低維高頻（變化快）、高維低頻（變化慢）

    PE = np.zeros((seq_len, d_model))
    for pos in range(seq_len):
        for i in range(d_model // 2):
            denom = 10000 ** (2 * i / d_model)
            PE[pos, 2 * i]     = np.sin(pos / denom)
            PE[pos, 2 * i + 1] = np.cos(pos / denom)
    return PE


def layer_norm(x, eps=1e-6):
    # 把每個 token 的向量正規化成均值=0、標準差=1
    # 防止數值在經過多層後爆炸或消失
    mean = x.mean(axis=-1, keepdims=True)
    std  = x.std(axis=-1, keepdims=True)
    return (x - mean) / (std + eps)


def feed_forward(x, d_ff):
    # 對每個 token 獨立做非線性特徵轉換
    # 升維（512→2048）→ ReLU → 降維（2048→512）
    # 升維讓低維空間裡線性不可分的特徵，到高維空間後可以被切開

    d_model = x.shape[-1]
    np.random.seed(1)
    W1 = np.random.randn(d_model, d_ff) * 0.01
    b1 = np.zeros(d_ff)
    W2 = np.random.randn(d_ff, d_model) * 0.01
    b2 = np.zeros(d_model)

    hidden = np.maximum(0, x @ W1 + b1)   # ReLU：負數歸零
    return hidden @ W2 + b2


def encoder_block(x, h, d_model, d_ff):
    # 完整的一層 Encoder
    # 兩個子層，每層都有 Residual Connection + LayerNorm

    # 子層 1：Multi-Head Self-Attention
    attn_out = multi_head_attention(x, h, d_model)
    x = layer_norm(x + attn_out)   # 加回原始 x（Residual），再正規化

    # 子層 2：Feed-Forward Network
    ffn_out = feed_forward(x, d_ff)
    x = layer_norm(x + ffn_out)    # 同上

    return x


def mini_transformer(x, N=6, h=8, d_model=512, d_ff=2048):
    # 完整的 Transformer Encoder

    seq_len = x.shape[0]

    # 加上位置資訊（讓模型知道每個 token 在哪個位置）
    x = x + positional_encoding(seq_len, d_model)

    # 重複 N 層 Encoder Block
    for layer in range(N):
        x = encoder_block(x, h, d_model, d_ff)
        print(f"  [✓] Encoder Block {layer + 1}/{N}，輸出 shape: {x.shape}")

    return x


# ============================================================
if __name__ == "__main__":
    np.random.seed(42)
    seq_len = 5
    d_model = 8    # 簡化版，論文是 512
    h       = 2    # 簡化版，論文是 8
    d_ff    = 16   # 簡化版，論文是 2048
    N       = 2    # 簡化版，論文是 6

    tokens = np.random.randn(seq_len, d_model)

    print("=" * 40)
    print("Mini Transformer")
    print("=" * 40)
    print(f"輸入 shape: {tokens.shape}")
    print(f"模型維度: {d_model}，頭數: {h}，層數: {N}")
    print()

    output = mini_transformer(tokens, N=N, h=h, d_model=d_model, d_ff=d_ff)

    print()
    print(f"輸出 shape: {output.shape}")
    print("輸入維度 == 輸出維度:", tokens.shape == output.shape)