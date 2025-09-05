import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# first run downloads
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

STOP = set(stopwords.words("english"))
ALPHA = re.compile(r"[^a-z]+")

def clean(text: str) -> str:
    text = text.lower()
    text = ALPHA.sub(" ", text)
    toks = [t for t in word_tokenize(text) if t not in STOP and len(t) > 1]
    return " ".join(toks)
