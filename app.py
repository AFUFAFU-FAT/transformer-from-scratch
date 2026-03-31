from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

app = Flask(__name__)
CORS(app)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用裝置：{device}")

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

model_name = "Qwen/Qwen2.5-7B-Instruct"
print("載入模型中...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map="auto"
)
model.eval()
print(f"模型載入完成！使用裝置：{device}")


def generate(prompt, max_tokens=200, temperature=0.7):
    messages = [
        {"role": "system", "content": "你是一個專門使用繁體中文回答的助手。無論任何情況，所有輸出必須使用繁體中文，嚴格禁止使用簡體中文。請用台灣習慣的用詞和語氣回答。"},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
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


@app.route("/generate", methods=["POST"])
def generate_endpoint():
    data = request.get_json()
    prompt = data.get("prompt", "")
    max_tokens = data.get("max_tokens", 200)
    temperature = data.get("temperature", 0.7)
    if not prompt:
        return jsonify({"error": "prompt is required"}), 400
    result = generate(prompt, max_tokens, temperature)
    return jsonify({"result": result})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "device": device})


if __name__ == "__main__":
    app.run(debug=False, port=5000)