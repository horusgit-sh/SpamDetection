import pandas as pd
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')
# Load dataset
def textPreparing():
    df = pd.read_csv(f"SMSSpamCollection", sep="\t", header=None, names=["label", "message"])
    df = df.drop_duplicates()

    df.info()


    df["message"] = df["message"].str.lower()

    df["message"] = df["message"].apply(lambda x: re.sub(r"[^a-z\s]", "", x))

    df["message"] = df["message"].apply(word_tokenize)
    stop_words = set(stopwords.words("english"))
    df["message"] = df["message"].apply(lambda x: [word for word in x if word not in stop_words])
    stemmer = PorterStemmer()
    df["message"] = df["message"].apply(lambda x: [stemmer.stem(word) for word in x])
    df["message"] = df["message"].apply(lambda x: " ".join(x))


    return df


def preprocess_message(message):

    message = message.lower()
    message = re.sub(r"[^a-z\s$!]", "", message)
    tokens = word_tokenize(message)
    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if word not in stop_words]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    return " ".join(tokens)


