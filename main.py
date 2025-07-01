import ir_measures
from ir_measures import MAP, MRR, P, Recall

from dataset.consts import FIRST_DATASET
from dataset.load import load_documents, load_qrels, load_queries
from dataset.process import process_token, tokenize_and_normalize, process

from services.tf_idf import generate_vectorizer, search as tf_idf_search


def generate_tfidf(dataset: str):
    docs = load_documents(dataset)
    docs_as_text = [doc.text for doc in docs]

    generate_vectorizer(
        docs_as_text,
        custom_tokenizer=tokenize_and_normalize,
        custom_preprocessor=process_token,
    )


def test_tfidf(dataset: str):
    docs = load_documents(dataset)
    queries = load_queries(dataset)
    qrels = load_qrels(dataset)

    run = []
    for query in queries:
        processd_query = process(query.text)
        processd_query = " ".join(processd_query)
        search_results = tf_idf_search(processd_query)

        for result in search_results:
            scoredDoc = ir_measures.ScoredDoc(
                query.query_id, docs[result["doc_idx"]].doc_id, result["score"]
            )
            run.append(scoredDoc)

    mesures = ir_measures.calc_aggregate(
        [MAP, Recall @ 100, P @ 10, MRR],
        qrels,
        run,
    )

    print(mesures)


def query_tfidf(query: str):
    docs = load_documents(FIRST_DATASET)

    processed_query = process(query)
    processed_query = " ".join(processed_query)

    results = tf_idf_search(processed_query)[:10]

    for result in results:
        doc_idx = result["doc_idx"]
        score = result["score"]
        doc = docs[doc_idx]
        print(doc)
        print(score)


def main():
    query_tfidf("How do I grow taller?")
    # generate_tfidf()
    # test_tfidf(FIRST_DATASET)


if __name__ == "__main__":
    main()
