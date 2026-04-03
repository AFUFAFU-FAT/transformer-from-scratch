import requests
import os
import json
import time

BASE_URL = "https://twtsl.ccu.edu.tw"
os.makedirs("videos", exist_ok=True)
os.makedirs("data", exist_ok=True)

def get_all_words(keywords):
    """搜尋多個關鍵字，收集所有詞彙的 id 和 name"""
    words = {}
    for kw in keywords:
        print(f"搜尋：{kw}")
        url = f"{BASE_URL}/api/manualSearch"
        params = {"name": kw, "lang": "zh", "page": 1, "pageSize": 50}
        try:
            res = requests.get(url, params=params, timeout=10)
            data = res.json()
            for record in data.get("Record", []):
                words[record["id"]] = record["name"]
            print(f"  找到 {len(data.get('Record', []))} 筆")
        except Exception as e:
            print(f"  錯誤: {e}")
        time.sleep(0.5)
    return words

def get_video_info(word_id):
    """取得單一詞彙的影片路徑和完整結構化特徵"""
    url = f"{BASE_URL}/api/querySearch"
    params = {"id": word_id, "lang": "zh"}
    try:
        res = requests.get(url, params=params, timeout=10)
        data = res.json()
        records = data.get("Record", [])
        if records:
            r = records[0]
            return {
                "clip":       r.get("clip"),
                "location1":  r.get("location1"),
                "location2":  r.get("location2"),
                "location3":  r.get("location3"),
                "handshape1": r.get("lo1_handshape_number_1"),
                "handshape2": r.get("lo1_handshape_number_2"),
                "hs1_name":   r.get("lo1_hs1"),
                "hs2_name":   r.get("lo1_hs2"),
                "stroke":     r.get("stroke"),
                "description": r.get("description"),
                "polysemy":   r.get("polysemy"),   # 多義詞編號
                "orders":     r.get("orders"),      # 排序
            }
    except Exception as e:
        print(f"  querySearch 錯誤: {e}")
    return None

def download_video(clip_path, save_name):
    """下載影片"""
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
    except Exception as e:
        print(f"  下載錯誤: {e}")
    return False

# ============================================================
# 搜尋的關鍵字清單（可自行擴充）
# ============================================================
keywords = [
    "你好", "謝謝", "再見", "是", "不是",
    "我", "你", "他", "我們", "喜歡",
    "吃", "喝", "去", "來", "家",
    "學校", "工作", "朋友", "今天", "明天",
    "幫忙", "需要", "可以", "不行", "好",
]

print("=== 開始爬取台灣手語辭典 ===")
words = get_all_words(keywords)
print(f"\n共找到 {len(words)} 個詞彙")

# 取得影片路徑、結構化特徵並下載
catalog = []
failed = []

for word_id, name in words.items():
    print(f"處理：{name} (id={word_id})")
    info = get_video_info(word_id)

    if info and info.get("clip"):
        save_name = f"{word_id}_{name}"
        success = download_video(info["clip"], save_name)
        if success:
            catalog.append({
                "id":          word_id,
                "name":        name,
                "video_file":  f"{save_name}.mp4",
                **info
            })
            print(f"  ✅ 下載成功")
        else:
            failed.append({"id": word_id, "name": name})
            print(f"  ❌ 下載失敗")
    else:
        failed.append({"id": word_id, "name": name})
        print(f"  ⚠️  無影片資料")

    time.sleep(0.3)

# 儲存完整目錄
with open("data/catalog.json", "w", encoding="utf-8") as f:
    json.dump(catalog, f, ensure_ascii=False, indent=2)

# 儲存失敗清單
if failed:
    with open("data/failed.json", "w", encoding="utf-8") as f:
        json.dump(failed, f, ensure_ascii=False, indent=2)

print(f"\n=== 完成 ===")
print(f"成功：{len(catalog)} 個")
print(f"失敗：{len(failed)} 個")
print(f"目錄已儲存到 data/catalog.json")