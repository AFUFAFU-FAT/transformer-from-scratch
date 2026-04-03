import requests
import json

# 先拿整個詞彙清單
url = "https://twtsl.ccu.edu.tw/api/manualSearch"
params = {
    "name": "",      # 空字串 → 搜尋全部
    "lang": "zh",
    "page": 1,
    "pageSize": 20
}

response = requests.get(url, params=params)
print("Status:", response.status_code)
data = response.json()
print(json.dumps(data, ensure_ascii=False, indent=2))