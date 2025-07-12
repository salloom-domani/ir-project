from typing import Any, Dict, Iterable, List, Optional

from langchain_community.retrievers import TFIDFRetriever
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from sklearn.feature_extraction.text import TfidfVectorizer

from utils.sanitize import sanitize

MAX_FEATURES = 100


class ScoredTFIDFRetriever(TFIDFRetriever):
    @classmethod
    def from_documents(
        cls,
        documents: Iterable[Document],
        *,
        tfidf_params: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> TFIDFRetriever:
        texts, metadatas = zip(*((d.page_content, d.metadata) for d in documents))

        tfidf_params = tfidf_params or {}
        vectorizer = TfidfVectorizer(**tfidf_params)
        tfidf_array = vectorizer.fit_transform(texts)
        metadatas = metadatas or ({} for _ in texts)
        docs = [
            Document(id=d.id, page_content=d.page_content, metadata=d.metadata)
            for d in documents
        ]
        return cls(vectorizer=vectorizer, docs=docs, tfidf_array=tfidf_array, **kwargs)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun, **kwargs: Any
    ) -> List[Document]:
        from sklearn.metrics.pairwise import cosine_similarity

        limit = kwargs.get("k", self.k)

        query_vec = self.vectorizer.transform([query])
        results = cosine_similarity(self.tfidf_array, query_vec).reshape((-1,))
        return_docs = [
            Document(
                id=self.docs[i].metadata["id"],
                page_content=self.docs[i].page_content,
                metadata={"score": results[i], "id": self.docs[i].metadata["id"]},
            )
            for i in results.argsort()[-limit:][::-1]
        ]
        return return_docs


def generate_vectorizer(
    dataset: str,
    docs: List[Document],
    max_features: int = MAX_FEATURES,
    custom_tokenizer=lambda x: x.split(),
    custom_preprocessor=lambda x: x,
):
    retriever = ScoredTFIDFRetriever.from_documents(
        docs,
        tfidf_params={
            "max_features": max_features,
            "tokenizer": custom_tokenizer,
            "preprocessor": custom_preprocessor,
        },
    )

    dataset = sanitize(dataset)
    retriever.save_local(
        ".objects",
        file_name=dataset,
    )

    return retriever


def generate_vectorizer_default(
    dataset: str,
    docs: List[Document],
):
    retriever = ScoredTFIDFRetriever.from_documents(
        docs,
    )

    dataset = sanitize(dataset)
    retriever.save_local(
        ".objects",
        file_name=f"default_{dataset}",
    )

    return retriever


def get_vectorizer(dataset: str):
    dataset = sanitize(dataset)
    retriever_copy = ScoredTFIDFRetriever.load_local(
        ".objects",
        file_name=dataset,
        allow_dangerous_deserialization=True,
    )
    return retriever_copy


def get_vectorizer_default(dataset: str):
    dataset = sanitize(dataset)
    retriever_copy = ScoredTFIDFRetriever.load_local(
        ".objects",
        file_name=f"default_{dataset}",
        allow_dangerous_deserialization=True,
    )
    return retriever_copy


def search(query: str, dataset: str, limit: int = 10):
    # vectorizer = get_vectorizer(dataset)
    vectorizer = get_vectorizer_default(dataset)
    results = vectorizer.invoke(query, None, k=limit)
    return results
