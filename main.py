from data_sets.second import load_dataset
from services.clean_text import apply_clean_tokenize_lemma_stem
from services import db_service

def main():
    print("Hello from ir-project!")

    # الاتصال بقاعدة البيانات
    conn = db_service.connect_db()

    # تحميل مجموعة البيانات (مثلاً 200,000 مستند)
    df_raw = load_dataset("beir/webis-touche2020", limit=200_000)

    # اقتصار البيانات على أول 5 أسطر فقط للاختبار
    df_raw = df_raw.head(5)

    print(df_raw.head())  # عرض أول 5 مستندات
    print(f"✅ عدد الوثائق المحملة: {len(df_raw)}")

    # ✅ 1) تخزين البيانات الخام في قاعدة البيانات
    raw_doc_ids = db_service.insert_raw_documents(conn, df_raw)
    print(f"✅ تم إدخال {len(raw_doc_ids)} وثيقة خام في قاعدة البيانات.")

    # ✅ 2) تنظيف ومعالجة البيانات (توكينايز، ليماتايزيشن، ستيم)
    df_processed = apply_clean_tokenize_lemma_stem(df_raw)
    print(df_processed.head())

    # ✅ 3) تخزين البيانات المعالجة في قاعدة البيانات مع ربطها بالوثائق الخام
    db_service.insert_processed_documents(conn, df_processed, raw_doc_ids)
    print("✅ تم إدخال الوثائق المعالجة في قاعدة البيانات.")

    # ✅ إغلاق الاتصال
    conn.close()
    print("✅ تم إغلاق الاتصال بقاعدة البيانات.")

if __name__ == "__main__":
    main()
