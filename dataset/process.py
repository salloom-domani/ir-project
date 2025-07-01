from nltk import pos_tag, download as download_nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize


TAG_DICT = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()


def download_nltk_data():
    download_nltk("stopwords")
    download_nltk("punkt")
    download_nltk("averaged_perceptron_tagger")
    download_nltk("averaged_perceptron_tagger_eng")
    download_nltk("wordnet")


def get_wordnet_pos(tag: str) -> str:
    return TAG_DICT.get(tag[0], wordnet.NOUN)


def process(text: str) -> list[str]:
    tokens = tokenize_and_normalize(text)
    tokens = lemmatization(tokens)
    # tokens = stemming(tokens)

    return tokens


def process_token(token: str) -> str:
    token = lemmatizer.lemmatize(token)

    return token


def tokenize_and_normalize(text: str) -> list[str]:
    tokens = word_tokenize(text)
    tokens = stopword_removal(tokens)
    return tokens


def stemming(tokens: list[str]) -> list[str]:
    return [stemmer.stem(token) for token in tokens]


def lemmatization(tokens: list[str]) -> list[str]:
    return [
        lemmatizer.lemmatize(token, get_wordnet_pos(tag))
        for token, tag in pos_tag(tokens)
    ]


def get_stop_words():
    return stopwords.words("english")


def stopword_removal(tokens: list[str]) -> list[str]:
    stop_words = set(stopwords.words("english"))
    return [token for token in tokens if token not in stop_words]
