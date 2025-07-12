from dataclasses import dataclass
from typing import List

from ir_datasets import load as load_dataset
from ir_measures import Qrel
from langchain_core.documents import Document


@dataclass
class Query:
    query_id: str
    text: str


def load_documents(name: str) -> List[Document]:
    dataset = load_dataset(name)
    return [
        Document(id=doc.doc_id, page_content=doc.text, metadata={"id": doc.doc_id})
        for doc in dataset.docs_iter()
    ]


def load_queries(name: str) -> list[Query]:
    dataset = load_dataset(name)
    return [
        Query(
            query_id=query.query_id,
            text=query.text,
        )
        for query in dataset.queries_iter()
    ]


def load_qrels(name: str) -> list[Qrel]:
    dataset = load_dataset(name)
    return [
        Qrel(qrel.query_id, qrel.doc_id, qrel.relevance)
        for qrel in dataset.qrels_iter()
    ]
