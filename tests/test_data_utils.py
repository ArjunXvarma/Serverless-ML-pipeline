# tests/test_data_utils.py

import pandas as pd
import numpy as np
from ml_pipeline.data_utils import TMDB_GENRES, make_train_test
from ml_pipeline.config import TrainingConfig

def test_make_train_test_shapes():
    # Minimal synthetic dataset
    df = pd.DataFrame(
        {
            "title": [
                "War Story",
                "Love in the City",
                "Mysterious Case",
                "Animal Friends",
            ],
            "overview": [
                "A heroic soldier fights in a great war.",
                "Two people fall in love in a small town.",
                "A detective investigates a mysterious crime.",
                "An animated adventure with talking animals.",
            ],
            # Use some known TMDB genre IDs
            "genre_ids": [
                [10752, 28],       # War, Action
                [10749, 18],       # Romance, Drama
                [9648, 80],        # Mystery, Crime
                [16, 12, 35],      # Animation, Adventure, Comedy
            ],
        }
    )

    cfg = TrainingConfig(test_size=0.25, random_state=42)
    X_train, X_test, Y_train, Y_test, genre_names = make_train_test(df, cfg)

    # Check non-empty splits
    assert len(X_train) > 0
    assert len(X_test) > 0
    assert Y_train.shape[0] == len(X_train)
    assert Y_test.shape[0] == len(X_test)

    # Multi-label: number of columns = number of genres in genre_names
    assert Y_train.shape[1] == len(genre_names)
    assert Y_test.shape[1] == len(genre_names)

    # Each row is multi-hot (0/1)
    assert set(np.unique(Y_train)).issubset({0, 1})
    assert set(np.unique(Y_test)).issubset({0, 1})


def test_tmdb_genres_contains_expected_ids():
    # Sanity check that common genre ids are present
    for gid in [28, 12, 16, 35, 80, 18]:
        assert gid in TMDB_GENRES, f"Expected TMDB_GENRES to contain id {gid}"
