import re
from nltk.tokenize import word_tokenize
import pandas as pd


def clean_and_tokenize(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = word_tokenize(text)
    return text, tokens


def apply_clean_and_tokenize(df):
    cleaned = df["text"].apply(clean_and_tokenize)
    df[["clean_text", "tokens"]] = pd.DataFrame(cleaned.tolist(), index=df.index)
    return df


# from load_data import load_dataset
# df = load_dataset("beir/trec-covid", 200_000)
# df = apply_clean_and_tokenize(df)
# print(df.head())
