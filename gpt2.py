import torch
import tiktoken
from transformers import GPT2LMHeadModel

# ============================================================
# GPT-2 文字生成
# 架構：載入 Hugging Face 的 GPT-2 預訓練權重
# 用 tiktoken 做 tokenization（和 GPT-2 原始實作一致）
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用裝置：{device}")

# Step 1：載入 tokenizer
enc = tiktoken.get_encoding("gpt2")

# Step 2：載入 GPT-2 預訓練模型
print("載入 GPT-2 模型...")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.to(device)
model.eval()
print("模型載入完成！")


def generate(prompt, max_tokens=50, temperature=0.8):
    """
    輸入提示詞，輸出生成的文字
    temperature 越高越有創意，越低越保守
    """
    # 把文字轉成 token ID
    token_ids = enc.encode(prompt)
    input_tensor = torch.tensor([token_ids], dtype=torch.long).to(device)

    with torch.no_grad():
        for _ in range(max_tokens):
            # 跑模型，取得下一個 token 的機率分布
            outputs = model(input_tensor)
            logits = outputs.logits[:, -1, :]  # 只看最後一個位置

            # 用 temperature 調整分布，再取樣
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # 把新 token 接到輸入後面
            input_tensor = torch.cat([input_tensor, next_token], dim=1)

            # 遇到結束符號就停
            if next_token.item() == enc.eot_token:
                break

    # 把 token ID 轉回文字
    generated_ids = input_tensor[0].tolist()
    return enc.decode(generated_ids)


# ============================================================
if __name__ == "__main__":
    prompt = "The future of AI is"
    print(f"\n提示詞：{prompt}")
    print("生成中...\n")

    result = generate(prompt, max_tokens=100, temperature=0.8)
    print("=" * 50)
    print(result)
    print("=" * 50)