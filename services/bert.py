from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from langchain_core.documents import Document
from langchain_text_splitters import SentenceTransformersTokenTextSplitter

from dataset.consts import FIRST_DATASET, SECOND_DATASET
from dataset.load import load_documents
from utils.sanitize import sanitize


def get_embedding_function():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


CHROMA_PATH = "chroma"

first_db = Chroma(
    persist_directory=CHROMA_PATH,
    collection_name=sanitize(FIRST_DATASET),
    embedding_function=get_embedding_function(),
)
second_db = Chroma(
    persist_directory=CHROMA_PATH,
    collection_name=sanitize(SECOND_DATASET),
    embedding_function=get_embedding_function(),
)

dbs = {
    FIRST_DATASET: first_db,
    SECOND_DATASET: second_db,
}


def search(query: str, dataset: str, limit: int = 10):
    db = dbs[dataset]
    results = db.similarity_search_with_score(query, k=limit)

    return results


def populate_chroma(dataset: str):
    documents = load_and_transform_documents(dataset)
    add_to_chroma(documents, dataset)


def load_and_transform_documents(dataset: str) -> list[Document]:
    documents = []
    docs = load_documents(dataset)

    for doc in docs:
        documents.append(
            Document(id=doc.id, page_content=doc.page_content, metadata={"id": doc.id})
        )
    return documents


def split_documents(documents: list[Document]):
    text_splitter = SentenceTransformersTokenTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )

    # Apply chunking per document
    chunks = []
    for doc in documents:
        chunks.extend(text_splitter.split_documents([doc]))
    return calculate_chunk_ids(chunks)


def add_to_chroma(chunks: list[Document], dataset: str):
    MAX_BATCH_SIZE = 4000  # Safe batch size (adjust as needed)
    collection_name = sanitize(dataset)

    db = Chroma(
        persist_directory=CHROMA_PATH,
        collection_name=collection_name,
        embedding_function=get_embedding_function(),
    )

    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    new_chunks = [chunk for chunk in chunks if chunk.metadata["id"] not in existing_ids]

    if not new_chunks:
        print("✅ No new documents to add")
        return

    print(f"👉 Adding {len(new_chunks)} new documents in batches")

    # Process in batches to avoid size limits
    for i in range(0, len(new_chunks), MAX_BATCH_SIZE):
        batch = new_chunks[i : i + MAX_BATCH_SIZE]
        batch_ids = [chunk.metadata["id"] for chunk in batch]

        print(
            f"   Adding batch {i // MAX_BATCH_SIZE + 1}/{(len(new_chunks) - 1) // MAX_BATCH_SIZE + 1} ({len(batch)} documents)"
        )

        db.add_documents(batch, ids=batch_ids)

    print("✅ All documents added successfully")


def calculate_chunk_ids(chunks: list[Document]):
    last_doc_id = None
    current_chunk_index = 0

    for chunk in chunks:
        doc_id = chunk.metadata["id"]
        if doc_id == last_doc_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0
        chunk.metadata["id"] = f"{doc_id}:{current_chunk_index}"
        last_doc_id = doc_id

    return chunks
