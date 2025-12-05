from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from .config import TrainingConfig

def build_genre_model(cfg: TrainingConfig) -> Pipeline:
    model = Pipeline(
        steps=[
            (
                'tfidf',
                TfidfVectorizer(
                    max_features=cfg.max_features,
                    ngram_range=(1, 2),
                    min_df=cfg.min_df,
                ),
            ),
            (
                'clf',
                OneVsRestClassifier(
                    LogisticRegression(max_iter=1000)
                ),
            )
        ]
    )

    return model