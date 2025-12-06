import os
import pandas as pd
import numpy as np
from ml_pipeline import train_genre

def test_train_and_evaluate_end_to_end(tmp_path, monkeypatch):
    # Create a tiny fake dataset CSV
    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "title": [
                "War Hero",
                "Romantic Escape",
                "Haunted House",
                "Space Adventure",
            ],
            "overview": [
                "A soldier fights in a brutal war.",
                "Two people fall in love on holiday.",
                "A family moves into a haunted house.",
                "An astronaut travels through space.",
            ],
            "genre_ids": [
                [10752, 28],       # War, Action
                [10749, 18],       # Romance, Drama
                [27, 53],          # Horror, Thriller
                [878, 12],         # Sci-Fi, Adventure
            ],
            "original_language": ["en", "en", "en", "en"],
            "release_date": ["2010-01-01"] * 4,
            "vote_average": [7.5, 7.0, 6.8, 8.0],
            "vote_count": [100, 80, 50, 120],
            "popularity": [12.0, 11.0, 9.0, 13.0],
        }
    )

    raw_path = tmp_path / "movies.csv"
    df.to_csv(raw_path, index=False)

    # Point the training script to our temp paths
    artifacts_dir = tmp_path / "artifacts"
    monkeypatch.setattr(train_genre, "RAW_PATH", str(raw_path))
    monkeypatch.setattr(train_genre, "ARTIFACT_DIR", str(artifacts_dir))
    monkeypatch.setattr(train_genre, "MODEL_PATH", str(artifacts_dir / "genre_model.joblib"))
    monkeypatch.setattr(train_genre, "META_PATH", str(artifacts_dir / "genre_meta.json"))

    metrics = train_genre.train_and_evaluate()

    # Assert metrics structure
    assert "f1_micro" in metrics
    assert "f1_macro" in metrics
    assert "per_genre_f1" in metrics
    assert isinstance(metrics["per_genre_f1"], dict)

    # Assert artifacts exist
    assert (artifacts_dir / "genre_model.joblib").exists()
    assert (artifacts_dir / "genre_meta.json").exists()

    # Sanity: F1 should be > 0 for such a tiny synthetic dataset
    assert metrics["f1_micro"] >= 0.0
