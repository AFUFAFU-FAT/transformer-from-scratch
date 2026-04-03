import requests

BASE_URL = "https://twtsl.ccu.edu.tw"

def get_word(word_id):
    url = f"{BASE_URL}/api/querySearch"
    params = {"id": word_id, "lang": "zh"}
    try:
        res = requests.get(url, params=params, timeout=5)
        data = res.json()
        records = data.get("Record", [])
        if records:
            return records[0].get("name")
    except:
        pass
    return None

# 測試更大的 ID
test_ids = [3500, 4000, 5000, 6000, 8000, 10000]
for i in test_ids:
    name = get_word(i)
    print(f"ID {i}: {name if name else '無資料'}")