import requests
from bs4 import BeautifulSoup
import re

API_KEY = "b006e9aa757fabeb334efec6a4afe24d9f8863d230a1f2ef02b9c9a638165075"

def get_vietnamese_location_name(query):
    """Tìm Wikipedia tiếng Việt và lấy tiêu đề bài viết (chính xác)."""
    try:
        # --- B1: Tìm link Wikipedia ---
        params = {
            "q": query + " site:vi.wikipedia.org",
            "hl": "vi",
            "gl": "vn",
            "api_key": API_KEY
        }
        res = requests.get("https://serpapi.com/search", params=params, timeout=10)
        data = res.json()

        wiki_url = None
        if "organic_results" in data:
            for r in data["organic_results"]:
                if "link" in r and "wikipedia.org" in r["link"]:
                    wiki_url = r["link"]
                    break

        if not wiki_url:
            return query

        # --- B2: Đọc tiêu đề thật từ trang Wikipedia ---
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept-Encoding": "gzip, deflate, br",
        }
        resp = requests.get(wiki_url, headers=headers, timeout=10)
        resp.encoding = resp.apparent_encoding

        soup = BeautifulSoup(resp.content.decode(resp.apparent_encoding, errors="ignore"), "html.parser")

        # Lấy tiêu đề chính thức trong <h1 id="firstHeading">
        heading = soup.find("h1", id="firstHeading")
        if heading and heading.text.strip():
            return heading.text.strip()

        # fallback dùng <title> nếu không tìm được h1
        title_tag = soup.find("title")
        if title_tag:
            clean = re.sub(r"\s*[-–—]\s*Wikipedia.*", "", title_tag.text).strip()
            return clean

        print(f"{query} → (Không đọc được tiêu đề) {wiki_url}")
        return query

    except Exception as e:
        print(f"Lỗi khi xử lý {query}: {e}")
        return query


# 🧠 Test
test_locations = [
    "Cau Vang",
    "Halong Bay",
    "Nha Tho Duc Ba",
    "Chua Thien Mu",
    "Phong Nha Cave",
]

for loc in test_locations:
    result = get_vietnamese_location_name(loc)
    print(f"{loc} → {result}")
