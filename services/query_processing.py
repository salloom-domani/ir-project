import numpy as np
import joblib
from services.clean_text import clean_text_for_tfidf
from services.bert_embeddings import compute_bert_embeddings
import pandas as pd

def process_query(query_text):
    """
    نظف الاستعلام ومثله باستخدام التمثيل الهجين (TF-IDF + BERT) ليتوافق مع الوثائق.
    """
    print("✅ بدء معالجة الاستعلام:", query_text)

    # 1) تنظيف الاستعلام
    clean_query = clean_text_for_tfidf(query_text)
    print("✅ النص بعد التنظيف:", clean_query)

    # 2) تمثيل الاستعلام باستخدام TF-IDF
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    query_tfidf = vectorizer.transform([clean_query]).toarray()
    print("✅ شكل تمثيل TF-IDF:", query_tfidf.shape)

    # 3) تمثيل الاستعلام باستخدام BERT
    df_query = pd.DataFrame([{"text": query_text}])
    query_bert = compute_bert_embeddings(df_query, text_col="text")
    print("✅ شكل تمثيل BERT:", query_bert.shape)

    # 4) دمج التمثيلين للحصول على التمثيل الهجين
    query_hybrid = np.hstack((query_tfidf, query_bert))
    print("✅ شكل التمثيل الهجين:", query_hybrid.shape)

    return query_hybrid
