"""
scrape_twsl.py
爬取「臺灣手語部編數位教材」18 冊所有單元的 Google Drive 影片連結

輸出：
  data/twsl_links.json   → 完整資料（冊、單元名稱、Drive 連結）
  data/twsl_links.csv    → 表格格式，方便查看
  data/twsl_links.txt    → 純連結清單

URL 格式：https://jung-hsingchang.tw/twsl/movies-uni{冊}.html
每頁包含該冊所有單元的 Google Drive 影片連結
"""

import requests
from bs4 import BeautifulSoup
import json
import csv
import time
import re
import os

BASE      = "https://jung-hsingchang.tw/twsl/"
TOTAL_BOOKS = 18
HEADERS   = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

OUTPUT_JSON = "data/twsl_links.json"
OUTPUT_CSV  = "data/twsl_links.csv"
OUTPUT_TXT  = "data/twsl_links.txt"


def fetch_page(url):
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        if r.status_code == 200:
            r.encoding = "utf-8"
            return BeautifulSoup(r.text, "html.parser")
        else:
            print(f"  HTTP {r.status_code}")
            return None
    except Exception as e:
        print(f"  錯誤：{e}")
        return None


def extract_drive_id(url):
    match = re.search(r'/file/d/([a-zA-Z0-9_-]+)', url)
    return match.group(1) if match else None


def parse_book_page(soup, book_num):
    units = []
    if not soup:
        return units

    tds = soup.find_all("td")
    for td in tds:
        # 同時支援 Google Drive 直連 和 reurl.cc 短網址
        a_tag = td.find("a", href=re.compile(r"(drive\.google\.com/file|reurl\.cc)"))
        if not a_tag:
            continue

        raw_url  = a_tag.get("href", "").split("?")[0]
        drive_id = extract_drive_id(raw_url)  # 短網址會是 None，沒關係

        # 提取單元名稱
        unit_name = td.get_text(separator="", strip=True)
        unit_name = re.sub(r'\s+', '', unit_name).strip()
        if not unit_name or unit_name == '&nbsp;':
            unit_name = f"第{len(units)+1}單元"

        units.append({
            "book":       book_num,
            "book_name":  f"第{book_num}冊",
            "unit":       len(units) + 1,
            "unit_name":  unit_name,
            "drive_id":   drive_id or "",
            "drive_url":  raw_url,
            "view_url":   f"https://drive.google.com/file/d/{drive_id}/view" if drive_id else raw_url,
        })

    return units

def main():
    os.makedirs("data", exist_ok=True)
    all_units = []

    print(f"🕷️  開始爬取台灣手語部編教材（共 {TOTAL_BOOKS} 冊）")
    print("=" * 60)

    for book in range(1, TOTAL_BOOKS + 1):
        url = f"{BASE}movies-uni{book}.html"
        print(f"\n📖 第 {book} 冊：{url}")

        soup = fetch_page(url)
        units = parse_book_page(soup, book)

        print(f"   找到 {len(units)} 個單元")
        for u in units:
            print(f"   {u['unit_name']} → {u['drive_id']}")
            all_units.append(u)

        time.sleep(0.8)  # 禮貌性延遲

    # ── 儲存 JSON ──
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_units, f, ensure_ascii=False, indent=2)

    # ── 儲存 CSV ──
    with open(OUTPUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["book_name", "unit_name", "view_url", "drive_id"])
        writer.writeheader()
        for u in all_units:
            writer.writerow({
                "book_name": u["book_name"],
                "unit_name": u["unit_name"],
                "view_url":  u["view_url"],
                "drive_id":  u["drive_id"],
            })

    # ── 儲存純連結 TXT ──
    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        for u in all_units:
            f.write(f"{u['book_name']} {u['unit_name']}\t{u['view_url']}\n")

    print(f"\n{'='*60}")
    print(f"✅ 完成！共 {len(all_units)} 個單元影片")
    print(f"   JSON：{OUTPUT_JSON}")
    print(f"   CSV ：{OUTPUT_CSV}")
    print(f"   TXT ：{OUTPUT_TXT}")


if __name__ == "__main__":
    main()