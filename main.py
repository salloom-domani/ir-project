from dataset.consts import FIRST_DATASET, SECOND_DATASET
from services.tfidf_langchain import generate_vectorizer_default
from dataset.load import load_documents


def main():
    first_dataset_docs = load_documents(FIRST_DATASET)
    generate_vectorizer_default(FIRST_DATASET, first_dataset_docs)

    second_dataset_docs = load_documents(SECOND_DATASET)
    generate_vectorizer_default(SECOND_DATASET, second_dataset_docs)


if __name__ == "__main__":
    main()
