from pymilvus import connections, Collection
from PIL import Image
from sentence_transformers import SentenceTransformer
import numpy as np
import requests
from io import BytesIO
from pathlib import Path
from bs4 import BeautifulSoup
import re

# ==============================
# 🔐 Zilliz Cloud connection
# ==============================
ZILLIZ_CLOUD_URI = "your_zilliz_cloud_uri_here"
ZILLIZ_API_KEY = " your_zilliz_api_key_here"

connections.connect(
    alias="default",
    uri=ZILLIZ_CLOUD_URI,
    token=ZILLIZ_API_KEY
)

# ==============================
# 🧠 Load CLIP model for image embeddings
# ==============================
encoder = SentenceTransformer('clip-ViT-B-32')

# ==============================
# 🧩 Helper: get image embedding
# ==============================
def get_image_embedding(image_name):
    """
    Load an image from the local 'input' directory and return its CLIP embedding.
    Example: get_image_embedding("CauVang.jpg")
    """
    try:
        # Construct full path inside 'input' folder
        image_path = Path("./input") / image_name

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Open and convert image to RGB
        image = Image.open(image_path).convert("RGB")

        # Generate embedding using CLIP
        embedding = encoder.encode([image], convert_to_numpy=True)[0]
        return np.array(embedding, dtype='float32')

    except Exception as e:
        raise RuntimeError(f"Failed to process image '{image_name}': {e}")
# ==============================
# 🔍 Search similar images in Zilliz
# ==============================
# API_KEY = "YOUR_SERPAPI_KEY_HERE"

# def get_vietnamese_location_name(query):
#     """Tìm Wikipedia tiếng Việt và lấy tiêu đề bài viết (chính xác)."""
#     try:
#         # --- B1: Tìm link Wikipedia ---
#         params = {
#             "q": query + " site:vi.wikipedia.org",
#             "hl": "vi",
#             "gl": "vn",
#             "api_key": API_KEY
#         }
#         res = requests.get("https://serpapi.com/search", params=params, timeout=10)
#         data = res.json()

#         wiki_url = None
#         if "organic_results" in data:
#             for r in data["organic_results"]:
#                 if "link" in r and "wikipedia.org" in r["link"]:
#                     wiki_url = r["link"]
#                     break

#         if not wiki_url:
#             return query

#         # --- B2: Đọc tiêu đề thật từ trang Wikipedia ---
#         headers = {
#             "User-Agent": "Mozilla/5.0",
#             "Accept-Encoding": "gzip, deflate, br",
#         }
#         resp = requests.get(wiki_url, headers=headers, timeout=10)
#         resp.encoding = resp.apparent_encoding

#         soup = BeautifulSoup(resp.content.decode(resp.apparent_encoding, errors="ignore"), "html.parser")

#         # Lấy tiêu đề chính thức trong <h1 id="firstHeading">
#         heading = soup.find("h1", id="firstHeading")
#         if heading and heading.text.strip():
#             return heading.text.strip()

#         # fallback: dùng <title>
#         title_tag = soup.find("title")
#         if title_tag:
#             clean = re.sub(r"\s*[-–—]\s*Wikipedia.*", "", title_tag.text).strip()
#             return clean

#         return query
#     except Exception:
#         return query

def search_similar_images(query_image, top_k=5):
    """Search for similar images from Zilliz given query image path or URL"""
    try:
        # Connect to existing collection
        collection = Collection("image_collection")
        collection.load()

        # Extract query embedding
        query_embedding = get_image_embedding(query_image)

        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}

        results = collection.search(
            data=[query_embedding.tolist()],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["image_path", "location"]
        )

        print(f"\n📸 Query Image: {query_image}")
        print("🖼️ Top similar images:\n" + "=" * 50)
        for hit in results[0]:
            img_path = hit.entity.get("image_path")
            location = hit.entity.get("location")
            similarity = 1 - hit.distance
            print(f"🔹 {img_path} | 📍 {location} | 📊 Similarity: {similarity:.2%}")

    except Exception as e:
        print(f"❌ Error: {e}")


# ==============================
# 🚀 Run Example
# ==============================
if __name__ == "__main__":
    # Example query — can be a local path or URL
    query_image = "skibidi.webp"
    search_similar_images(query_image, top_k=5)

    connections.disconnect("default")
