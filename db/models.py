from sqlmodel import Field, SQLModel, Relationship


class Dataset(SQLModel, table=True):
    name: str = Field(primary_key=True)
    docs: list["Document"] = Relationship(back_populates="dataset", cascade_delete=True)

    def __repr__(self) -> str:
        return f"Dataset(name={self.name!r}, docs={len(self.docs)!r})"


class Document(SQLModel, table=True):
    id: str = Field(primary_key=True)
    content: str = Field()
    dataset_name: str = Field(foreign_key="dataset.name")
    dataset: Dataset = Relationship(back_populates="docs")

    def __repr__(self) -> str:
        return f"Document(id={self.id!r}, content={self.content!r})"
