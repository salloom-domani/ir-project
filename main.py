from dataset.consts import FIRST_DATASET, SECOND_DATASET
from dataset.load import load_documents, load_qrels, load_queries


def main():
    docs = load_documents(SECOND_DATASET, 2)
    print(docs)

    queries = load_queries(FIRST_DATASET)
    print(queries)

    qrels = load_qrels(SECOND_DATASET)
    print(qrels)


if __name__ == "__main__":
    main()
