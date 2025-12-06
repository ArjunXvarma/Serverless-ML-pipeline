from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
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
                    LinearSVC(C=1.0)
                ),
            )
        ]
    )

    return model