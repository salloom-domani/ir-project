from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
import joblib
import pandas as pd


def compute_tfidf(
    df: pd.DataFrame, raw_col: str = "text", clean_func=None, max_features: int = 10000
):
    if clean_func is not None:
        cleaned_texts = df[raw_col].apply(clean_func)
    else:
        cleaned_texts = df[raw_col]

    vectorizer = TfidfVectorizer(max_features=max_features)
    tfidf_matrix = vectorizer.fit_transform(cleaned_texts)

    joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
    sparse.save_npz("tfidf_matrix.npz", tfidf_matrix)

    return vectorizer, tfidf_matrix
