from data_sets.second import load_dataset
from services.clean_text import apply_clean_tokenize_lemma_stem, clean_text_for_tfidf
from services import db_service
from services.tfidf import compute_tfidf
from services.bert_embeddings import compute_bert_embeddings, save_bert_embeddings

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

if __name__ == "__main__":
    # شغّل دالة معينة حسب الحاجة:
    
    # لتشغيل كامل الخطوات من تحميل وتخزين وتنظيف
    # run_full_pipeline()
    
    # لتشغيل فقط TF-IDF على البيانات الخام المخزنة في قاعدة البيانات
    # fetch_raw_from_db_and_tfidf()
    
    fetch_raw_from_db_and_bert_embeddings()