from ir_measures import ScoredDoc, calc_aggregate
from ir_measures import MAP, MRR, P, Recall

from dataset.load import load_qrels, load_queries
from dataset.process import process

from services.tfidf import get_vectorizer_default as get_vectorizer


def test_tfidf_langchain(dataset: str):
    queries = load_queries(dataset)
    qrels = load_qrels(dataset)
    vectorizer = get_vectorizer(dataset)

    run = []
    for query in queries:
        processd_query = process(query.text)
        processd_query = " ".join(processd_query)
        search_results = vectorizer.invoke(processd_query)

        for doc in search_results:
            scoredDoc = ScoredDoc(query.query_id, doc.id or "", doc.metadata["score"])
            run.append(scoredDoc)

    mesures = calc_aggregate(
        [MAP, Recall @ 100, P @ 10, MRR],
        qrels,
        run,
    )

    print(mesures)
