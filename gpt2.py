import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ============================================================
# Qwen2.5-7B-Instruct + 4-bit 量化
# 繁體中文支援，8GB VRAM 可跑
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用裝置：{device}")

# 4-bit 量化設定
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

model_name = "Qwen/Qwen2.5-7B-Instruct"
print("載入模型中（首次需下載約 14GB）...")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map="auto"
)
model.eval()
print("模型載入完成！")


def generate(prompt, max_tokens=200, temperature=0.7):
    messages = [
        {"role": "system", "content": "你是一個專門使用繁體中文回答的助手。無論任何情況，所有輸出必須使用繁體中文，嚴格禁止使用簡體中文。請用台灣習慣的用詞和語氣回答。"},
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer([text], return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    generated = outputs[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


if __name__ == "__main__":
    prompt = "人工智慧的未來是什麼？"
    print(f"\n提示詞：{prompt}")
    print("生成中...\n")
    result = generate(prompt, max_tokens=200)
    print("=" * 50)
    print(result)
    print("=" * 50)