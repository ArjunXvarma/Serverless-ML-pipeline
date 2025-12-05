import os
import json
import joblib
from sklearn.metrics import f1_score
from .config import TrainingConfig
from .data_utils import load_raw_movies, make_train_test
from .model_genre import build_genre_model

RAW_PATH = "data/raw/movies.csv"
ARTIFACT_DIR = "artifacts"
MODEL_PATH = os.path.join(ARTIFACT_DIR, "genre_model.joblib")
META_PATH = os.path.join(ARTIFACT_DIR, "genre_meta.json")

def train_and_evaluate():
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    cfg = TrainingConfig()

    print(f"Loading data from {RAW_PATH}...")
    df = load_raw_movies(RAW_PATH)
    print(f"Loaded {len(df)} movies.")

    X_train, X_test, Y_train, Y_test, genre_names = make_train_test(df, cfg)

    model = build_genre_model(cfg)
    print("Training model...")
    model.fit(X_train, Y_train)

    print("Evaluating...")
    Y_pred = model.predict(X_test)

    f1_micro = f1_score(Y_test, Y_pred, average="micro", zero_division=0)
    f1_macro = f1_score(Y_test, Y_pred, average="macro", zero_division=0)

    metrics = {
        "f1_micro": float(f1_micro),
        "f1_macro": float(f1_macro),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }

    print("Saving model and metadata...")
    joblib.dump(model, MODEL_PATH)

    meta = {
        "genre_names": genre_names,
        "metrics": metrics,
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    print("Done. Metrics:", metrics)
    return metrics


if __name__ == "__main__":
    train_and_evaluate()
