from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

@dataclass
class ModelConfig:
    max_features: int = 50000
    ngram_range: tuple = (1,2)
    C: float = 1.0

def build_baseline(cfg: ModelConfig = ModelConfig()) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(max_features=cfg.max_features,
                                  ngram_range=cfg.ngram_range,
                                  min_df=2, max_df=0.9)),
        ("clf", LinearSVC(C=cfg.C))
    ])
