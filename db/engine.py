from sqlmodel import Session, SQLModel, create_engine

from db.models import Dataset, Document

from dataset.consts import FIRST_DATASET, SECOND_DATASET
from dataset.load import load_documents


connect_args = {"check_same_thread": False}
engine = create_engine("sqlite:///db.sqlite", connect_args=connect_args)


def create_db_and_tables():
    SQLModel.metadata.create_all(engine)


def populate_db():
    first_docs = load_documents(FIRST_DATASET)
    second_docs = load_documents(SECOND_DATASET)

    with Session(engine) as session:
        first_dataset = Dataset(
            name=FIRST_DATASET,
            docs=[
                Document(
                    id=doc.id or "",
                    content=doc.page_content,
                    dataset_name=FIRST_DATASET,
                )
                for doc in first_docs
            ],
        )
        second_dataset = Dataset(
            name=SECOND_DATASET,
            docs=[
                Document(
                    id=doc.id or "",
                    content=doc.page_content,
                    dataset_name=SECOND_DATASET,
                )
                for doc in second_docs
            ],
        )
        session.add_all([first_dataset, second_dataset])
        session.commit()
