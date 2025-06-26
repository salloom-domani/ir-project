import re
from nltk.tokenize import word_tokenize
import pandas as pd

def clean_and_tokenize(text):
    """تنظيف وتجهيز النص للتعامل معه."""
    text = text.lower()  # تحويل الأحرف إلى صغيرة
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)  # إزالة الروابط
    text = re.sub(r"[^\w\s]", '', text)  # إزالة الرموز وعلامات الترقيم
    text = re.sub(r"\d+", '', text)  # إزالة الأرقام
    text = re.sub(r"\s+", ' ', text).strip()  # إزالة المسافات الزائدة
    tokens = word_tokenize(text)  # تقسيم النص إلى كلمات
    return text, tokens

def apply_clean_and_tokenize(df):
    """تطبيق التنظيف والتجزئة على عمود text وإرجاع DataFrame جديد."""
    cleaned = df["text"].apply(clean_and_tokenize)
    df[["clean_text", "tokens"]] = pd.DataFrame(cleaned.tolist(), index=df.index)
    return df

# مثال على الاستخدام:
# from load_data import load_dataset
# df = load_dataset("beir/trec-covid", 200_000)
# df = apply_clean_and_tokenize(df)
# print(df.head())
