from sqlalchemy import create_engine
from sqlalchemy import select
from sqlalchemy.orm import Session

from dataset.consts import FIRST_DATASET, SECOND_DATASET
from dataset.load import load_documents


from .models import Base, Dataset, Document


engine = create_engine("sqlite:///db.sqlite")


def populate_db():
    Base.metadata.create_all(engine)
    first_docs = load_documents(FIRST_DATASET)
    second_docs = load_documents(SECOND_DATASET)

    with Session(engine) as session:
        first_dataset = Dataset(
            name=FIRST_DATASET,
            docs=[Document(id=doc.id, content=doc.page_content) for doc in first_docs],
        )
        second_dataset = Dataset(
            name=SECOND_DATASET,
            docs=[Document(id=doc.id, content=doc.page_content) for doc in second_docs],
        )
        session.add_all([first_dataset, second_dataset])
        session.commit()


def get_all_documents(dataset: str, limit=None):
    with Session(engine) as session:
        stmt = select(Document).where(Document.dataset_name.is_(dataset)).limit(limit)
        docs = session.scalars(stmt).all()
        return docs


def get_document(doc_id: str):
    with Session(engine) as session:
        stmt = select(Document).where(Document.id.is_(doc_id))
        doc = session.scalar(stmt)
        return doc
