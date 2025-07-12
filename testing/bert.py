from ir_measures import ScoredDoc, calc_aggregate
from ir_measures import MAP, MRR, P, Recall

from dataset.load import load_qrels, load_queries
from services.bert import search as bert_search


def test_bert(dataset: str):
    queries = load_queries(dataset)
    qrels = load_qrels(dataset)

    run = []
    for query in queries:
        # processd_query = process(query.text)
        # processd_query = " ".join(processd_query)
        search_results = bert_search(query.text, dataset)

        for doc, score in search_results:
            scoredDoc = ScoredDoc(query.query_id, doc.id or "", score)
            run.append(scoredDoc)

    mesures = calc_aggregate(
        [MAP, Recall @ 100, P @ 10, MRR],
        qrels,
        run,
    )

    print(mesures)
