from transformers import BertTokenizer, BertModel
import torch
import numpy as np
import pandas as pd

# تحميل الموديل والمحول (Tokenizer)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding='max_length')
    with torch.no_grad():
        outputs = model(**inputs)
    cls_embedding = outputs.last_hidden_state[:,0,:].squeeze().numpy()
    return cls_embedding

def compute_bert_embeddings(df: pd.DataFrame, text_col='text'):
    print("Start computing BERT embeddings...")
    embeddings = df[text_col].apply(get_bert_embedding)
    emb_matrix = np.vstack(embeddings.values)
    print(f"BERT embeddings shape: {emb_matrix.shape}")
    return emb_matrix

def save_bert_embeddings(emb_matrix, file_path="bert_embeddings.npy"):
    np.save(file_path, emb_matrix)
    print(f"BERT embeddings saved to {file_path}")

def load_bert_embeddings(file_path="bert_embeddings.npy"):
    emb_matrix = np.load(file_path)
    print(f"BERT embeddings loaded from {file_path}")
    return emb_matrix
