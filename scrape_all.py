import requests
import os
import json
import time

BASE_URL = "https://twtsl.ccu.edu.tw"
os.makedirs("videos", exist_ok=True)
os.makedirs("data", exist_ok=True)

def get_video_info(word_id):
    url = f"{BASE_URL}/api/querySearch"
    params = {"id": word_id, "lang": "zh"}
    try:
        res = requests.get(url, params=params, timeout=10)
        data = res.json()
        records = data.get("Record", [])
        if records:
            r = records[0]
            return {
                "id":          word_id,
                "name":        r.get("name"),
                "clip":        r.get("clip"),
                "location1":   r.get("location1"),
                "location2":   r.get("location2"),
                "handshape1":  r.get("lo1_handshape_number_1"),
                "hs1_name":    r.get("lo1_hs1"),
                "stroke":      r.get("stroke"),
                "description": r.get("description"),
            }
    except:
        pass
    return None

def download_video(clip_path, save_name):
    video_url = f"{BASE_URL}/{clip_path}.mp4"
    save_path = f"videos/{save_name}.mp4"
    if os.path.exists(save_path):
        return True
    try:
        res = requests.get(video_url, timeout=15, stream=True)
        if res.status_code in [200, 206]:
            with open(save_path, "wb") as f:
                for chunk in res.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True
    except:
        pass
    return False

# ============================================================
# 從 ID 1 爬到 3500，跳過已下載的
# ============================================================
catalog = []
failed_ids = []

# 載入已有的目錄（支援斷點續傳）
catalog_path = "data/catalog_full.json"
done_ids = set()
if os.path.exists(catalog_path):
    with open(catalog_path, encoding="utf-8") as f:
        catalog = json.load(f)
        done_ids = {item["id"] for item in catalog}
    print(f"已有 {len(catalog)} 筆，從斷點繼續...")

ID_START = 1
ID_END = 3700
SAVE_EVERY = 50  # 每 50 筆存一次

print(f"=== 開始爬取 ID {ID_START} ~ {ID_END} ===")

for word_id in range(ID_START, ID_END + 1):
    if word_id in done_ids:
        continue

    info = get_video_info(word_id)

    if info and info.get("clip") and info.get("name"):
        save_name = f"{word_id}_{info['name']}"
        success = download_video(info["clip"], save_name)
        if success:
            info["video_file"] = f"{save_name}.mp4"
            catalog.append(info)
            print(f"✅ [{word_id}] {info['name']}")
        else:
            failed_ids.append(word_id)
            print(f"❌ [{word_id}] 下載失敗")
    else:
        print(f"⬜ [{word_id}] 無資料")

    # 定期儲存（斷點續傳）
    if word_id % SAVE_EVERY == 0:
        with open(catalog_path, "w", encoding="utf-8") as f:
            json.dump(catalog, f, ensure_ascii=False, indent=2)

    time.sleep(0.2)

# 最終儲存
with open(catalog_path, "w", encoding="utf-8") as f:
    json.dump(catalog, f, ensure_ascii=False, indent=2)

print(f"\n=== 完成 ===")
print(f"成功：{len(catalog)} 個詞彙")
print(f"失敗：{len(failed_ids)} 個")
print(f"目錄：data/catalog_full.json")