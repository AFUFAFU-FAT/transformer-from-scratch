import requests
import json

url = "https://twtsl.ccu.edu.tw/api/manualSearch"
params = {
    "name": "你好",
    "lang": "zh",
    "page": 1,
    "pageSize": 10
}

response = requests.get(url, params=params)
data = response.json()
print(json.dumps(data, ensure_ascii=False, indent=2))