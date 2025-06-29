import psycopg2
from psycopg2.extras import execute_values
import pandas as pd

def connect_db():
    conn = psycopg2.connect(
        dbname="ir_db",
        user="ir_user",
        password="ir_user",
        host="localhost"
    )
    return conn

def insert_raw_documents(conn, df):
    tuples = list(df[['doc_id', 'text']].itertuples(index=False, name=None))
    cursor = conn.cursor()
    query = "INSERT INTO raw_documents (title, text) VALUES %s RETURNING doc_id;"
    execute_values(cursor, query, tuples)
    doc_ids = cursor.fetchall()
    conn.commit()
    cursor.close()
    return [doc_id[0] for doc_id in doc_ids]

def insert_processed_documents(conn, processed_df, raw_doc_ids):
    tuples = []
    for idx, row in processed_df.iterrows():
        tuples.append((
            raw_doc_ids[idx],
            row['clean_text'],
            row['lemmas'],
            row['stems']
        ))
    cursor = conn.cursor()
    query = """INSERT INTO processed_documents (raw_doc_id, clean_text, lemmas, stems) VALUES %s"""
    execute_values(cursor, query, tuples)
    conn.commit()
    cursor.close()

def fetch_raw_documents(conn, limit=None):
    cursor = conn.cursor()
    query = "SELECT doc_id, text FROM raw_documents"
    if limit:
        query += f" LIMIT {limit}"
    cursor.execute(query)
    rows = cursor.fetchall()
    cursor.close()
    df = pd.DataFrame(rows, columns=["doc_id", "text"])
    return df


def fetch_raw_documents(conn, limit=5):
    query = f"SELECT doc_id, text FROM raw_documents LIMIT {limit};"
    cursor = conn.cursor()
    cursor.execute(query)
    rows = cursor.fetchall()
    cursor.close()
    # تحويل النتائج إلى DataFrame
    df = pd.DataFrame(rows, columns=["doc_id", "text"])
    return df

def insert_hybrid_embeddings(conn, raw_doc_ids, hybrid_matrix):
    tuples = []
    for idx, doc_id in enumerate(raw_doc_ids):
        embedding = hybrid_matrix[idx].tolist()  # حول التمثيل إلى قائمة
        tuples.append((doc_id, embedding))
    
    cursor = conn.cursor()
    query = """INSERT INTO hybrid_embeddings (raw_doc_id, embedding) VALUES %s"""
    execute_values(cursor, query, tuples)
    conn.commit()
    cursor.close()
    print(f"✅ تم إدخال {len(raw_doc_ids)} تمثيلًا هجينيًا في قاعدة البيانات.")