import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import wordnet, stopwords
import pandas as pd

# تحميل الموارد المطلوبة لمرة واحدة فقط
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))  # هنا قائمة الكلمات التوقيفية للإنجليزية

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def clean_tokenize_lemma_stem(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token not in stop_words]

    pos_tags = nltk.pos_tag(tokens)
    lemmas = [lemmatizer.lemmatize(token, get_wordnet_pos(pos)) for token, pos in pos_tags]
    stems = [stemmer.stem(token) for token in tokens]

    # ✅ النص النهائي: يمكنك الاختيار أيهما تريد تخزينه كنص أساسي
    clean_text = " ".join(lemmas)  # لو أردت نصًا يعتمد على lemmatization
    # clean_text = " ".join(stems)  # أو هذا لو أردت نصًا يعتمد على stemming

    return clean_text, lemmas, stems


def apply_clean_tokenize_lemma_stem(df):
    print("start cleaning, removing stopwords, lemmatization, and stemming")
    cleaned = df["text"].apply(clean_tokenize_lemma_stem)
    df[["clean_text", "lemmas", "stems"]] = pd.DataFrame(cleaned.tolist(), index=df.index)
    print("end processing")
    return df

def clean_text_for_tfidf(text):
    clean_text, _, _ = clean_tokenize_lemma_stem(text)
    return clean_text