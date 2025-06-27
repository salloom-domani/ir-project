import ir_datasets
import pandas as pd


def load_dataset(dataset="antique", limit=200_000):
    ds = ir_datasets.load(dataset)
    docs_iter = ds.docs_iter()

    cleaned_docs = []
    while len(cleaned_docs) < limit:
        try:
            doc = next(docs_iter)
            cleaned_docs.append({"doc_id": doc.doc_id, "text": doc.text})
        except (UnicodeDecodeError, RuntimeError):
            continue
        except StopIteration:
            break

    df = pd.DataFrame(cleaned_docs)
    print(f"✅ عدد الوثائق المحملة: {df.shape[0]}")
    return df
