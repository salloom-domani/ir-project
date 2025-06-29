from ir_datasets import load as load_dataset
from dataclasses import dataclass


@dataclass
class Document:
    doc_id: str
    text: str


@dataclass
class Query:
    query_id: str
    text: str


@dataclass
class Qrel:
    query_id: str
    doc_id: str
    relevance: int
    iteration: str


def load_documents(name: str) -> list[Document]:
    dataset = load_dataset(name)
    return [Document(doc_id=doc.doc_id, text=doc.text) for doc in dataset.docs_iter()]


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
        Qrel(
            query_id=qrel.query_id,
            doc_id=qrel.doc_id,
            relevance=qrel.relevance,
            iteration=qrel.iteration,
        )
        for qrel in dataset.qrels_iter()
    ]
