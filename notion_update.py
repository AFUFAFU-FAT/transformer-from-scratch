"""
寫入今日開發進度至 Notion 專案開發紀錄
"""
import urllib.request, json

import os
TOKEN   = os.environ.get("NOTION_TOKEN", "")
PAGE_ID = "3357b8a7-470d-811d-a19f-cf6b37b90833"

HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json",
    "Notion-Version": "2022-06-28",
}

def h2(text):
    return {"object":"block","type":"heading_2","heading_2":{"rich_text":[{"type":"text","text":{"content":text}}]}}

def h3(text):
    return {"object":"block","type":"heading_3","heading_3":{"rich_text":[{"type":"text","text":{"content":text}}]}}

def p(text):
    return {"object":"block","type":"paragraph","paragraph":{"rich_text":[{"type":"text","text":{"content":text}}]}}

def bullet(text):
    return {"object":"block","type":"bulleted_list_item","bulleted_list_item":{"rich_text":[{"type":"text","text":{"content":text}}]}}

def divider():
    return {"object":"block","type":"divider","divider":{}}

blocks = [
    divider(),
    h2("2026-04-05　辨識精準度優化與長句輸出支援"),

    h3("🔴 問題一：「是」長期被誤判為「說」"),
    p("實機比出「是」（五指全伸、靜止於下巴），系統卻持續輸出「說」。"),
    bullet("診斷：是 和 說 binary 幾乎相同，模型信心偏低（約 10%）但仍排在候選清單中"),
    bullet("問題根源一：是 的判定門檻 11%，信心 10% 被擋掉，差一點點"),
    bullet("問題根源二：apply_feature_gates 中「是」的靜止 gate（wrist_std < 0.12）在靠近下巴時可能因手腕微動而失效"),
    bullet("問題根源三：是 的專項偵測有 all5 and static 兩個 guard，moving 動作導致 static=False 未觸發"),

    h3("✅ 解法：是 專項偵測重構（最高優先機制）"),
    bullet("移除 final_lbl != '是' 限制：無論其他路徑選了什麼，每次都查「是」的機率"),
    bullet("移除 all5 and static guard：改為無條件查模型原始機率，不設手形/動作篩選"),
    bullet("新增雙來源觸發：候選清單（candidates top3）中「是」≥ 8% 或 直查模型「是」≥ 5% → 強制輸出「是」"),
    bullet("是 一般路徑門檻：11% → 8%（對應候選清單直接選中的情況）"),
    bullet("是偵測 bypass 門檻：5%（直查模型機率路徑）"),
    bullet("效果：是 被說覆蓋的情況完全消除，實機測試正確識別"),

    h3("🟡 問題二：單字判定正確但無法輸出長句子"),
    p("word_buffer 顯示只有最後 5 個單字，無法輸出完整句子。"),
    bullet("診斷：UI 顯示使用 word_buffer[-5:] 截斷，buffer 本身沒有上限"),
    bullet("解法：移除 [-5:] 截斷，改為每行 8 個自動換行，顯示全部詞彙"),
    bullet("候選清單（合併/序列/結尾）動態跟在詞彙行下方，不固定 y 座標"),
    bullet("效果：可比出完整長句，所有歷史詞彙保留在畫面上"),

    h3("🔵 檔案整理"),
    bullet("診斷工具統一移至 tools/ 資料夾：check_binary.py / check_confusion.py / check_feat.py / check_position.py / check_samples.py / check_zhao_conf.py / count_label.py / delete_label.py"),
    bullet("recognize.py 標記為舊版（已由 recognize_endpose.py 取代，保留供參考）"),
    bullet("output.json / status.sh / recognize_error.log 加入 .gitignore（資源監控用，不入版本控制）"),

    h3("📊 目前辨識狀態（2026-04-05）"),
    bullet("32 詞彙全部可正確識別，實機測試通過"),
    bullet("是/找/再見/喜歡 四個難辨識詞彙均已有專項處理"),
    bullet("是：專項偵測 5%/8% 雙門檻，最高優先"),
    bullet("找：Gate 4（食指彎+中無小伸+無左手）→ ZhaoGate 強制替換"),
    bullet("再見：wrist_x_std > 0.06 動作 gate + 自適應 endpose 權重（<0.30 時 75/25）"),
    bullet("喜歡：(wrist_x_std > 0.05 or wrist_y_std > 0.05) 動作 gate + 門檻 11%"),
]

body = json.dumps({"children": blocks}).encode("utf-8")
req  = urllib.request.Request(
    f"https://api.notion.com/v1/blocks/{PAGE_ID}/children",
    data=body, headers=HEADERS, method="PATCH"
)
try:
    with urllib.request.urlopen(req) as r:
        print(f"✅ 成功寫入 Notion（狀態碼 {r.status}）")
except urllib.error.HTTPError as e:
    print(f"❌ 失敗：{e.code} {e.reason}")
    print(e.read().decode())
