from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
import joblib
import pandas as pd

def compute_tfidf(
    df: pd.DataFrame, 
    raw_col: str = "text", 
    clean_func = None,
    max_features: int = 10000
):
    """
    تنظيف النصوص باستخدام دالة وتنفيذ حساب TF-IDF وحفظ النموذج.

    Parameters:
        df: DataFrame يحتوي النصوص الخام
        raw_col: اسم عمود النص الخام داخل df
        clean_func: دالة تأخذ نصًا وتُرجع نصًا منظفًا
        max_features: أقصى عدد من الميزات في الـ TF-IشيDF

    Returns:
        vectorizer: نموذج TF-IDF المدرب
        tfidf_matrix: مصفوفة TF-IDF
    """
    # إذا تم تمرير دالة تنظيف، نطبقها على كل نص
    if clean_func is not None:
        cleaned_texts = df[raw_col].apply(clean_func)
    else:
        cleaned_texts = df[raw_col]

    vectorizer = TfidfVectorizer(max_features=max_features)
    tfidf_matrix = vectorizer.fit_transform(cleaned_texts)

    # حفظ النموذج والمصفوفة
    joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
    sparse.save_npz("tfidf_matrix.npz", tfidf_matrix)

    return vectorizer, tfidf_matrix
