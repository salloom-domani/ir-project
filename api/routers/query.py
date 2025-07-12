from fastapi import APIRouter
from langchain_core.documents import Document

from db.engine import FIRST_DATASET
from services.bert import search as search_using_bert
from services.tfidf import search as search_using_tfidf
from dataset.process import process as process_query


router = APIRouter()


@router.get("/tfidf")
def search_tfidf(
    query: str,
    dataset: str = FIRST_DATASET,
    limit: int = 10,
) -> list[Document]:
    results = search_using_tfidf(query, dataset, limit)
    return results


@router.get("/bert")
def search_bert(
    query: str,
    dataset: str = FIRST_DATASET,
    limit: int = 10,
) -> list[Document]:
    results = search_using_bert(query, dataset, limit)
    return [
        Document(
            id=result[0].id,
            page_content=result[0].page_content,
            metadata={"score": result[1]},
        )
        for result in results
    ][::-1]


@router.get("/process")
def process_query_string(query: str) -> list[str]:
    tokens = process_query(query)
    return tokens
