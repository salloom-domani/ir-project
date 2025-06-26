import ir_datasets
import pandas as pd
from itertools import islice


def load_dataset(dataset_id="antique", limit=200_000):
    """
    تحميل مجموعة بيانات باستخدام ir_datasets وتحويلها إلى DataFrame.
    Args:
        dataset_id (str): معرف مجموعة البيانات، مثلاً "antique" أو "beir/trec-covid"
        limit (int): عدد الوثائق التي تريد تحميلها
    Returns:
        pd.DataFrame: البيانات المحملة (doc_id, text)
    """
    ds = ir_datasets.load("antique")
    docs_iter = ds.docs_iter()

    cleaned_docs = []
    while len(cleaned_docs) < limit:
        try:
            doc = next(docs_iter)
            cleaned_docs.append({
                "doc_id": doc.doc_id,
                "text": doc.text
            })
        except (UnicodeDecodeError, RuntimeError):
            continue
        except StopIteration:
            break

    df = pd.DataFrame(cleaned_docs)
    print(f"✅ عدد الوثائق المحملة: {df.shape[0]}")
    return df
