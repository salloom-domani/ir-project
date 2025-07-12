from typing import Annotated, Sequence
from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select

from db.engine import FIRST_DATASET, engine
from db.models import Document


def get_session():
    with Session(engine) as session:
        yield session


SessionDep = Annotated[Session, Depends(get_session)]
router = APIRouter()


@router.get("/")
def read_documents(
    session: SessionDep,
    dataset: str = FIRST_DATASET,
    skip: int = 0,
    limit: int = 10,
) -> Sequence[Document]:
    documents = session.exec(
        select(Document)
        .where(Document.dataset_name == dataset)
        .offset(skip)
        .limit(limit)
    ).all()
    return documents


@router.get("/{doc_id}")
def read_document(doc_id: str, session: SessionDep) -> Document:
    document = session.get(Document, doc_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    return document
