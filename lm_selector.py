"""
lm_selector.py — LLM 斷點評分器

從候選詞清單中選出最符合台灣手語語序的詞。
LLM 只能「選」，不能「改」。
"""

import re
import anthropic

# 32 個詞彙庫（必須與訓練集一致）
VOCAB_32 = [
    "一樣", "不喜歡", "不是", "人", "介紹", "他", "住", "你",
    "再見", "問", "喜歡", "嗎", "地方", "好", "媽媽", "學",
    "家", "對不起", "想", "我", "我們", "找", "是", "晚上",
    "爸爸", "現在", "甚麼", "聽人", "認真", "說", "謝謝", "開心",
]

LLM_TIMEOUT = 2.0   # 秒，超時就 fallback
LLM_ENABLED = False  # 設為 False 停用 LLM，永遠取最高信心度候選


def lm_select(candidates: list, word_buffer: list) -> str:
    """
    從候選清單中選出最符合語序的詞。

    Parameters
    ----------
    candidates : list of dict
        每個元素含 {"key": "A"/"B"/"C", "label": str, "conf": float}
    word_buffer : list of str
        已確認輸出的前文詞彙（不可被修改）

    Returns
    -------
    str
        選中的詞彙 label；若 LLM 失敗則 fallback 取最高信心度
    """
    if not LLM_ENABLED:
        return max(candidates, key=lambda c: c["conf"])["label"]

    client = anthropic.Anthropic()

    buffer_str = "（無）" if not word_buffer else " ".join(word_buffer)
    keys = [c["key"] for c in candidates]
    candidates_str = "\n".join(
        f"  {c['key']}. {c['label']}（{c['conf']*100:.0f}%）" for c in candidates
    )
    vocab_str = "、".join(VOCAB_32)
    keys_str = "/".join(keys)

    prompt = (
        f"你是台灣手語詞彙斷點判斷器。\n"
        f"已辨識的前文詞彙：{buffer_str}  ← 這些是確定的，不可修改\n"
        f"目前這個位置的候選詞（含信心度）：\n"
        f"{candidates_str}\n"
        f"字彙庫（只能從這 32 個詞選）：{vocab_str}\n\n"
        f"請從 {keys_str} 中選一個最符合台灣手語語序的選項。\n"
        f"只輸出字母（{'、'.join(keys)}），不做其他修改。"
    )

    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=5,
            messages=[{"role": "user", "content": prompt}],
            timeout=LLM_TIMEOUT,
        )
        raw = response.content[0].text.strip().upper()
        match = re.search(r"[A-C]", raw)
        if match:
            chosen_key = match.group()
            for c in candidates:
                if c["key"] == chosen_key:
                    return c["label"]
    except Exception as e:
        print(f"  [LM] 呼叫失敗，fallback: {e}")

    # Fallback：信心度最高的候選
    return max(candidates, key=lambda c: c["conf"])["label"]
