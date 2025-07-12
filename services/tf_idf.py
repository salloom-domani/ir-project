from os import makedirs
from joblib import dump, load
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# number of terms to keep in the vocabulary
MAX_FEATURES = 10000


def generate_vectorizer(
    dataset: str,
    docs: list[str],
    max_features: int = MAX_FEATURES,
    custom_tokenizer=lambda x: x.split(),
    custom_preprocessor=lambda x: x,
):
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        tokenizer=custom_tokenizer,
        preprocessor=custom_preprocessor,
    )
    tfidf_matrix = vectorizer.fit_transform(docs)

    makedirs(".objects", exist_ok=True)

    dump(vectorizer, ".objects/tfidf_vectorizer.joblib")
    dump(tfidf_matrix, ".objects/tfidf_matrix.joblib")

    return vectorizer, tfidf_matrix


def get_vectorizer(dataset: str):
    vectorizer = load(".objects/tfidf_vectorizer.joblib")
    tfidf_matrix = load(".objects/tfidf_matrix.joblib")

    return vectorizer, tfidf_matrix


# def search(query: str, dataset: str):
#     vectorizer, tfidf_matrix = get_vectorizer(dataset)
#     query_vec = vectorizer.transform([query])
#     scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
#
#     ranked_indices = scores.argsort()[::-1]
#
#     results = []
#     for rank, idx in enumerate(ranked_indices):
#         results.append({"score": scores[idx], "doc_idx": idx, "rank": rank + 1})
#
#     return results


def search(query: str, vectorizer, tfidf_matrix):
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()

    ranked_indices = scores.argsort()[::-1]

    results = []
    for rank, idx in enumerate(ranked_indices):
        results.append({"score": scores[idx], "doc_idx": idx, "rank": rank + 1})

    return results
