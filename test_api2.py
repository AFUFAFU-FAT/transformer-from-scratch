import requests
import json

# 看單一詞彙的詳細資料
url = "https://twtsl.ccu.edu.tw/api/querySearch"
params = {"id": 2387, "lang": "zh"}

response = requests.get(url, params=params)
print(json.dumps(response.json(), ensure_ascii=False, indent=2))