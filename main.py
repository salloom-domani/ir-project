# def query_tfidf(query: str, dataset):
#     docs = load_documents(dataset)
#
#     processed_query = process(query)
#     processed_query = " ".join(processed_query)
#
#     results = tf_idf_search(processed_query, dataset)[:10]
#
#     for result in results:
#         doc_idx = result["doc_idx"]
#         score = result["score"]
#         doc = docs[doc_idx]
#         print(doc)
#         print(score)
#


def main():
    query = "How do I grow taller?"
    # query_tfidf("How do I grow taller?")
    # generate_tfidf(SECOND_DATASET)
    # test_tfidf(SECOND_DATASET)
    # populate_chroma(FIRST_DATASET)
    # populate_chroma(SECOND_DATASET)
    # resutls = bert_search(query, FIRST_DATASET)
    # print(resutls)

    # test_bert(FIRST_DATASET)
    # test_bert(SECOND_DATASET)
    # generate_tfidf_langchain(FIRST_DATASET)
    # test_tfidf(FIRST_DATASET)
    # test_tfidf_langchain(FIRST_DATASET)

    # results = tf_idf_search_lang(query, FIRST_DATASET)
    # for result in results:
    #     print(result)

    # populate_db()
    # docs = get_all_documents(FIRST_DATASET, 10)
    # print(docs)


if __name__ == "__main__":
    main()
