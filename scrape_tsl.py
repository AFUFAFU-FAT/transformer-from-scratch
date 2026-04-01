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

def get_video_path(word_id):
    """取得單一詞彙的影片路徑"""
    url = f"{BASE_URL}/api/querySearch"
    params = {"id": word_id, "lang": "zh"}
    try:
        res = requests.get(url, params=params, timeout=10)
        data = res.json()
        records = data.get("Record", [])
        if records and records[0].get("clip"):
            return records[0]["clip"]
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

# 先從常用詞彙開始測試
keywords = [
    "你好", "謝謝", "再見", "是", "不是",
    "我", "你", "他", "我們", "喜歡",
    "吃", "喝", "去", "來", "家"
]

print("=== 開始爬取台灣手語辭典 ===")
words = get_all_words(keywords)
print(f"\n共找到 {len(words)} 個詞彙")

# 取得影片路徑並下載
catalog = []
for word_id, name in words.items():
    print(f"處理：{name} (id={word_id})")
    clip = get_video_path(word_id)
    if clip:
        save_name = f"{word_id}_{name}"
        success = download_video(clip, save_name)
        if success:
            catalog.append({"id": word_id, "name": name, "clip": clip})
            print(f"  ✅ 下載成功")
        else:
            print(f"  ❌ 下載失敗")
    time.sleep(0.3)

# 儲存目錄
with open("data/catalog.json", "w", encoding="utf-8") as f:
    json.dump(catalog, f, ensure_ascii=False, indent=2)

print(f"\n完成！共下載 {len(catalog)} 個影片")
print(f"目錄已儲存到 data/catalog.json")