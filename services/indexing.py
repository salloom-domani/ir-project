import faiss
import numpy as np

def build_and_save_faiss_index(embeddings_path="hybrid_embeddings.npy", index_path="faiss_index.index"):
    print("تحميل التمثيلات الهجينة...")
    embeddings = np.load(embeddings_path).astype(np.float32)
    print("شكل التمثيلات:", embeddings.shape)

    print("تحميل ترتيب doc_ids...")
    doc_ids = np.load("doc_id_order.npy")
    print(f"✅ عدد doc_ids المحملة: {len(doc_ids)}")

    # اختيار IndexFlatIP (inner product) أو IndexFlatL2
    index = faiss.IndexFlatIP(embeddings.shape[1])
    print("بناء الفهرس...")
    index.add(embeddings)
    print(f"✅ تمت إضافة {index.ntotal} تمثيلًا إلى الفهرس.")

    print(f"حفظ الفهرس في {index_path}...")
    faiss.write_index(index, index_path)
    print("✅ تم حفظ الفهرس بنجاح.")

    # حفظ ترتيب doc_ids في ملف مرتبط بالفهرس
    np.save("faiss_doc_id_order.npy", doc_ids)
    print("✅ تم حفظ ترتيب doc_ids في faiss_doc_id_order.npy")
