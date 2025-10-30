import streamlit as st
from PIL import Image
import os
from pymilvus import Collection
from query_data import get_image_embedding
import numpy as np
from collections import defaultdict

st.set_page_config(page_title="Phân loại ảnh theo địa điểm", layout="wide")
st.title("🌏 Phân loại ảnh theo địa điểm (Zilliz + CLIP)")

uploaded_files = st.file_uploader(
    "📤 Tải lên nhiều ảnh (jpg, jpeg, png, webp):",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True
)

if uploaded_files:
    st.info("⏳ Đang xử lý và truy vấn từng ảnh, vui lòng chờ...")

    input_dir = "input"
    os.makedirs(input_dir, exist_ok=True)

    # Kết nối đến Zilliz
    collection = Collection("image_collection")
    collection.load()

    grouped_results = defaultdict(list)  # {location: [ (image, similarity) ]}

    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)
        filename = uploaded_file.name
        temp_path = os.path.join(input_dir, filename)
        image.save(temp_path)

        try:
            query_embedding = get_image_embedding(filename)
            search_params = {"metric_type": "L2", "params": {"nprobe": 10}}

            results = collection.search(
                data=[query_embedding.tolist()],
                anns_field="embedding",
                param=search_params,
                limit=1,  # chỉ lấy top 1
                output_fields=["image_path", "location"]
            )

            if results and len(results[0]) > 0:
                hit = results[0][0]
                location = hit.entity.get("location")
                similarity = (1 - hit.distance) * 100
                grouped_results[location].append((image, similarity))
            else:
                grouped_results["Không xác định"].append((image, 0))

        except Exception as e:
            st.error(f"❌ Lỗi khi xử lý ảnh {filename}: {e}")

        finally:
            # Xóa ảnh tạm
            if os.path.exists(temp_path):
                os.remove(temp_path)

    # ======= HIỂN THỊ KẾT QUẢ =======
    if grouped_results:
        st.success("✅ Đã hoàn tất phân loại ảnh theo địa điểm!")

        for location, images in grouped_results.items():
            st.markdown(f"### 📍 **{location}** ({len(images)} ảnh)")
            cols = st.columns(4)
            for idx, (img, sim) in enumerate(images):
                with cols[idx % 4]:
                    st.image(img, caption=f"{sim:.2f}%", use_container_width=True)
    else:
        st.warning("⚠️ Không có ảnh nào được phân loại thành công.")
