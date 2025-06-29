import faiss
import numpy as np

def build_and_save_faiss_index(embeddings_path="hybrid_embeddings.npy", index_path="faiss_index.index"):
    print("تحميل التمثيلات الهجينة...")
    embeddings = np.load(embeddings_path).astype(np.float32)
    print("شكل التمثيلات:", embeddings.shape)

    # اختيار IndexFlatIP (inner product) أو IndexFlatL2
    index = faiss.IndexFlatIP(embeddings.shape[1])
    print("بناء الفهرس...")
    index.add(embeddings)
    print(f"✅ تمت إضافة {index.ntotal} تمثيلًا إلى الفهرس.")

    print(f"حفظ الفهرس في {index_path}...")
    faiss.write_index(index, index_path)
    print("✅ تم حفظ الفهرس بنجاح.")
