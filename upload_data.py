from pymilvus import connections, Collection, DataType, FieldSchema, CollectionSchema, utility
from pathlib import Path
from PIL import Image
import numpy as np
from sentence_transformers import SentenceTransformer
import re

# ==============================
# 🔐 Zilliz Cloud connection
# ==============================
ZILLIZ_CLOUD_URI = "https://in03-975f9d29d0053ed.serverless.aws-eu-central-1.cloud.zilliz.com"
ZILLIZ_API_KEY = "45afcdb4720c03aedd93bd6e38a86da54d296f193eb5e64350cba4468f2ffcd6eb08fc6da2b6ed9213165662e4ab1482cb23363d"

connections.connect(alias="default", uri=ZILLIZ_CLOUD_URI, token=ZILLIZ_API_KEY)

# ==============================
# 🧠 CLIP model
# ==============================
encoder = SentenceTransformer('clip-ViT-B-32')  # 512-dim

# ==============================
# 🏗️ Create or load collection
# ==============================
def create_image_collection():
    if utility.has_collection("image_collection"):
        print("📁 Collection already exists.")
        return Collection("image_collection")

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),  # manual IDs for upsert
        FieldSchema(name="image_path", dtype=DataType.VARCHAR, max_length=255),
        FieldSchema(name="location", dtype=DataType.VARCHAR, max_length=255),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=512)
    ]
    schema = CollectionSchema(fields, description="Image embeddings collection")
    collection = Collection("image_collection", schema)

    index_params = {"metric_type": "L2", "index_type": "IVF_FLAT", "params": {"nlist": 1024}}
    collection.create_index(field_name="embedding", index_params=index_params)
    print("✅ Created new collection 'image_collection'")
    return collection

# ==============================
# 🧩 Extract embeddings
# ==============================
def extract_image_features(image_path):
    image = Image.open(image_path).convert("RGB")
    embedding = encoder.encode([image], convert_to_numpy=True)[0]
    return np.array(embedding, dtype='float32')

# ==============================
# 🔄 Upsert images
# ==============================
def upsert_image(collection, image_id, image_path, location, embedding):
    # Normalize path for Milvus query
    safe_path = image_path.replace("\\", "/")

    # Check if record exists
    expr = f'image_path == "{safe_path}"'
    results = collection.query(expr=expr, output_fields=["id"])

    if results:
        old_id = results[0]["id"]
        print(f"🟡 Updating existing record for {safe_path} (id={old_id})")
        collection.delete(expr=f"id == {old_id}")

    data = [
        [image_id],
        [safe_path],
        [location],
        [embedding.tolist()]
    ]

    collection.upsert(data)
    print(f"✅ Upserted {safe_path}")

collection = create_image_collection()

images_dir = Path('./images')
image_files = [f for f in images_dir.iterdir() if f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.jfif')]

if not image_files:
    print("❌ No images found in ./images folder.")
    exit()

for idx, image_path in enumerate(image_files, start=1):
    try:
        # Extract location from filename
        location = image_path.stem.strip()

        # Delete trailing digits if any
        location = re.sub(r"\d+$", "", location).strip()

        vec = extract_image_features(str(image_path))

        # Upsert data to collection
        upsert_image(collection, idx, str(image_path), location, vec)

    except Exception as e:
        print(f"❌ Error processing {image_path.name}: {e}")

collection.flush()
collection.load()
print("\n✅ All images upserted successfully!")
connections.disconnect("default")