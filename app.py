from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import tiktoken
from transformers import GPT2LMHeadModel

app = Flask(__name__)
CORS(app)

# 啟動時載入模型（只載入一次）
device = "cuda" if torch.cuda.is_available() else "cpu"
enc = tiktoken.get_encoding("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.to(device)
model.eval()
print(f"模型載入完成，使用裝置：{device}")


def generate(prompt, max_tokens=100, temperature=0.8):
    token_ids = enc.encode(prompt)
    input_tensor = torch.tensor([token_ids], dtype=torch.long).to(device)

    with torch.no_grad():
        for _ in range(max_tokens):
            outputs = model(input_tensor)
            logits = outputs.logits[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_tensor = torch.cat([input_tensor, next_token], dim=1)
            if next_token.item() == enc.eot_token:
                break

    return enc.decode(input_tensor[0].tolist())


@app.route("/generate", methods=["POST"])
def generate_endpoint():
    data = request.get_json()
    prompt = data.get("prompt", "")
    max_tokens = data.get("max_tokens", 100)
    temperature = data.get("temperature", 0.8)

    if not prompt:
        return jsonify({"error": "prompt is required"}), 400

    result = generate(prompt, max_tokens, temperature)
    return jsonify({"result": result})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "device": device})


if __name__ == "__main__":
    app.run(debug=False, port=5000)