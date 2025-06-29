from data_sets.second import load_dataset
from services.clean_text import apply_clean_tokenize_lemma_stem, clean_text_for_tfidf
from services import db_service
from services.tfidf import compute_tfidf
from services.bert_embeddings import compute_bert_embeddings, save_bert_embeddings
from services.hybrid import compute_hybrid_embeddings
from services.indexing import build_and_save_faiss_index
from services.query_processing import process_query
import faiss
import numpy as np
from services.query_matching import search_and_rank


def connect_db():
    print("الاتصال بقاعدة البيانات...")
    return db_service.connect_db()

def load_data(limit=5):
    print(f"تحميل مجموعة بيانات بحد {limit} مستندات...")
    df = load_dataset("beir/webis-touche2020", limit=limit)
    print(df.head())
    return df

def insert_raw_data(conn, df_raw):
    print("إدخال البيانات الخام في قاعدة البيانات...")
    raw_doc_ids = db_service.insert_raw_documents(conn, df_raw)
    print(f"تم إدخال {len(raw_doc_ids)} وثيقة خام في قاعدة البيانات.")
    return raw_doc_ids

def process_and_insert_clean_data(conn, df_raw, raw_doc_ids):
    print("تنظيف ومعالجة البيانات...")
    df_processed = apply_clean_tokenize_lemma_stem(df_raw)
    print(df_processed.head())

    print("إدخال البيانات المعالجة في قاعدة البيانات...")
    db_service.insert_processed_documents(conn, df_processed, raw_doc_ids)
    print("تم إدخال البيانات المعالجة في قاعدة البيانات.")

def run_full_pipeline():
    conn = connect_db()
    df_raw = load_data(limit=5)
    raw_doc_ids = insert_raw_data(conn, df_raw)
    process_and_insert_clean_data(conn, df_raw, raw_doc_ids)
    conn.close()
    print("تم إغلاق الاتصال بقاعدة البيانات.")

def fetch_raw_from_db_and_tfidf(limit=5):
    conn = connect_db()
    print("جلب البيانات الخام من قاعدة البيانات...")
    df_raw = db_service.fetch_raw_documents(conn, limit=limit)
    print(df_raw.head())

    print("تشغيل TF-IDF على البيانات الخام مع تطبيق دالة التنظيف...")
    vectorizer, tfidf_matrix = compute_tfidf(df_raw, raw_col="text", clean_func=clean_text_for_tfidf, max_features=10000)
    print("شكل مصفوفة TF-IDF:", tfidf_matrix.shape)
    print("أمثلة على بعض الكلمات:", vectorizer.get_feature_names_out()[:10])

    conn.close()
    print("تم إغلاق الاتصال بقاعدة البيانات بعد TF-IDF.")

def fetch_raw_from_db_and_bert_embeddings(limit=5):
    conn = connect_db()
    print("جلب البيانات الخام من قاعدة البيانات...")
    df_raw = db_service.fetch_raw_documents(conn, limit=limit)
    print(df_raw.head())

    print("تشغيل BERT embeddings على البيانات الخام...")
    emb_matrix = compute_bert_embeddings(df_raw, text_col="text")

    save_bert_embeddings(emb_matrix, file_path="bert_embeddings.npy")

    conn.close()
    print("تم إغلاق الاتصال بقاعدة البيانات بعد BERT embeddings.")

def compute_and_save_hybrid_embeddings_and_insert_to_db():
    conn = connect_db()
    print("جلب البيانات الخام من قاعدة البيانات...")
    df_raw = db_service.fetch_raw_documents(conn, limit=5)
    raw_doc_ids = df_raw['doc_id'].tolist()
    print(df_raw.head())

    print("بدء حساب التمثيل الهجين...")
    hybrid_matrix = compute_hybrid_embeddings(
        tfidf_path="tfidf_matrix.npz",
        bert_path="bert_embeddings.npy",
        save_path="hybrid_embeddings.npy"
    )
    print("✅ تم حساب التمثيل الهجين.")

    print("بدء إدخال التمثيل الهجين في قاعدة البيانات...")
    db_service.insert_hybrid_embeddings(conn, raw_doc_ids, hybrid_matrix)

    np.save("doc_id_order.npy", np.array(raw_doc_ids))
    print("✅ تم حفظ ترتيب doc_ids في doc_id_order.npy")


    conn.close()
    print("✅ تم إغلاق الاتصال بقاعدة البيانات بعد إدخال التمثيل الهجين.")




def build_faiss_index():
    print("بدء بناء الفهرس باستخدام FAISS...")
    build_and_save_faiss_index(
        embeddings_path="hybrid_embeddings.npy",
        index_path="faiss_index.index"
    )
    print("✅ تم بناء الفهرس وحفظه.")



def process_and_search_query(query_text, top_k=5):
    """
    معالجة الاستعلام والبحث في الفهرس المبني مسبقًا مع إظهار doc_ids الحقيقية.
    """
    # 1) معالجة الاستعلام لإنتاج التمثيل الهجين
    query_hybrid = process_query(query_text).astype('float32')

    # 2) تحميل الفهرس
    print("✅ تحميل الفهرس من faiss_index.index...")
    index = faiss.read_index("faiss_index.index")

    # 3) تحميل doc_ids المقابلة للفهرس
    print("✅ تحميل doc_ids المرتبطة بالفهرس...")
    doc_ids = np.load("faiss_doc_id_order.npy")

    # 4) البحث في الفهرس
    print("✅ بدء البحث في الفهرس...")
    distances, indices = index.search(query_hybrid, top_k)

    print("✅ نتائج البحث:")
    for rank, (idx, dist) in enumerate(zip(indices[0], distances[0]), 1):
        real_doc_id = doc_ids[idx]  # تحويل الفهرس داخل faiss إلى doc_id حقيقي
        print(f"[{rank}] Document ID: {real_doc_id}, Distance: {dist}")


if __name__ == "__main__":
    # شغّل دالة معينة حسب الحاجة:
    
    # لتشغيل كامل الخطوات من تحميل وتخزين وتنظيف
    # run_full_pipeline()
    
    # لتشغيل فقط TF-IDF على البيانات الخام المخزنة في قاعدة البيانات
    # fetch_raw_from_db_and_tfidf()
    
    # fetch_raw_from_db_and_bert_embeddings()

    # لتشغيل التمثيل الهجين
    # compute_and_save_hybrid_embeddings_and_insert_to_db()

    # لتشغيل الفهرسة:
    # build_faiss_index()

    # لمعالجة استعلام معين والبحث عنه
    # process_and_search_query("What you learn in university")

    query = "What do you learn in university?"
    search_and_rank(query)
