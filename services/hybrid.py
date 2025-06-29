import numpy as np
from scipy import sparse
from psycopg2.extras import execute_values
def compute_hybrid_embeddings(tfidf_path="tfidf_matrix.npz", bert_path="bert_embeddings.npy", save_path="hybrid_embeddings.npy"):
    print("تحميل مصفوفة TF-IDF...")
    tfidf_matrix = sparse.load_npz(tfidf_path).toarray()  # حولها إلى numpy array
    print("شكل TF-IDF:", tfidf_matrix.shape)

    print("تحميل مصفوفة BERT...")
    bert_matrix = np.load(bert_path)
    print("شكل BERT:", bert_matrix.shape)

    if tfidf_matrix.shape[0] != bert_matrix.shape[0]:
        raise ValueError("عدد المستندات في TF-IDF وBERT غير متطابق!")

    print("دمج التمثيلين بشكل تسلسلي...")
    hybrid_matrix = np.hstack((tfidf_matrix, bert_matrix))
    print("شكل التمثيل الهجين:", hybrid_matrix.shape)

    np.save(save_path, hybrid_matrix)
    print(f"تم حفظ التمثيل الهجين في {save_path}")

    return hybrid_matrix

def insert_hybrid_embeddings(conn, raw_doc_ids, hybrid_matrix):
    tuples = []
    for idx, doc_id in enumerate(raw_doc_ids):
        # احصل على التمثيل الهجين للمستند idx
        embedding = hybrid_matrix[idx].tolist()  # حوله لقائمة ليدعمه psycopg2
        tuples.append((doc_id, embedding))
    
    cursor = conn.cursor()
    query = """INSERT INTO hybrid_embeddings (raw_doc_id, embedding) VALUES %s"""
    execute_values(cursor, query, tuples)
    conn.commit()
    cursor.close()
    print(f"✅ تم إدخال {len(raw_doc_ids)} تمثيلًا هجينيًا في قاعدة البيانات.")