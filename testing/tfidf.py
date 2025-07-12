from ir_measures import ScoredDoc, calc_aggregate
from ir_measures import MAP, MRR, P, Recall

from dataset.load import load_documents, load_qrels, load_queries
from dataset.process import process

from services.tf_idf import search, get_vectorizer


def test_tfidf(dataset: str):
    docs = load_documents(dataset)
    queries = load_queries(dataset)
    qrels = load_qrels(dataset)

    vectorizer, tfidf_matrix = get_vectorizer(dataset)

    run = []
    for query in queries:
        processd_query = process(query.text)
        processd_query = " ".join(processd_query)
        search_results = search(processd_query, vectorizer, tfidf_matrix)

        for result in search_results:
            scoredDoc = ScoredDoc(
                query.query_id, docs[result["doc_idx"]].doc_id, result["score"]
            )
            run.append(scoredDoc)

    mesures = calc_aggregate(
        [MAP, Recall @ 100, P @ 10, MRR],
        qrels,
        run,
    )

    print(mesures)
