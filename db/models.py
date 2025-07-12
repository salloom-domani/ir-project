from typing import List
from sqlalchemy import ForeignKey
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.orm import relationship


class Base(DeclarativeBase):
    pass


class Dataset(Base):
    __tablename__ = "dataset"
    name: Mapped[str] = mapped_column(primary_key=True)
    docs: Mapped[List["Document"]] = relationship(
        back_populates="dataset", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"Dataset(name={self.name!r}, docs={len(self.docs)!r})"


class Document(Base):
    __tablename__ = "document"
    id: Mapped[str] = mapped_column(primary_key=True)
    content: Mapped[str]
    dataset_name: Mapped[str] = mapped_column(ForeignKey("dataset.name"))
    dataset: Mapped["Dataset"] = relationship(back_populates="docs")

    def __repr__(self) -> str:
        return f"Document(id={self.id!r}, content={self.content!r})"
