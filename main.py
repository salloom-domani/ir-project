from dataset.consts import FIRST_DATASET, SECOND_DATASET
from services.tfidf import generate_vectorizer_default
from dataset.load import load_documents

from testing.bert import test_bert
from testing.tfidf import test_tfidf


def main():
    # first_dataset_docs = load_documents(FIRST_DATASET)
    # generate_vectorizer_default(FIRST_DATASET, first_dataset_docs)
    #
    # second_dataset_docs = load_documents(SECOND_DATASET)
    # generate_vectorizer_default(SECOND_DATASET, second_dataset_docs)
    test_bert(FIRST_DATASET)
    # test_tfidf(FIRST_DATASET)


if __name__ == "__main__":
    main()
