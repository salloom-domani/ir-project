import numpy as np
import faiss
from sklearn.metrics.pairwise import cosine_similarity
from services.query_processing import process_query

def load_resources():
    print("تحميل الموارد اللازمة للبحث...")
    index = faiss.read_index("faiss_index.index")
    doc_ids = np.load("faiss_doc_id_order.npy")
    hybrid_embeddings = np.load("hybrid_embeddings.npy").astype(np.float32)
    print(f"تم تحميل الفهرس وعدد الوثائق: {len(doc_ids)}")
    return index, doc_ids, hybrid_embeddings

def search_and_rank(query_text, top_k=5):
    print(f"معالجة الاستعلام: {query_text}")
    query_embedding = process_query(query_text).astype(np.float32)

    index, doc_ids, hybrid_embeddings = load_resources()

    distances, indices = index.search(query_embedding, top_k)
    retrieved_doc_ids = doc_ids[indices[0]]
    retrieved_embeddings = hybrid_embeddings[indices[0]]

    cos_sim = cosine_similarity(query_embedding, retrieved_embeddings).flatten()
    sorted_idx = np.argsort(-cos_sim)

    print("النتائج بعد ترتيبها حسب التشابه:")
    for rank, idx in enumerate(sorted_idx, 1):
        doc_id = retrieved_doc_ids[idx]
        score = cos_sim[idx]
        print(f"{rank}. Document ID: {doc_id}, Cosine Similarity: {score:.4f}")
