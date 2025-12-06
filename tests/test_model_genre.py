from ml_pipeline.model_genre import build_genre_model
from ml_pipeline.config import TrainingConfig
import numpy as np

def test_build_genre_model_and_fit():
    cfg = TrainingConfig(max_features=1000, min_df=1)
    model = build_genre_model(cfg)

    texts = [
        "A brave hero saves the world in an epic battle.",
        "A couple falls in love in a small romantic town.",
        "A group of friends must survive a terrifying monster.",
    ]

    # Multi-label targets: 3 samples x 2 genres (e.g. Action, Romance)
    Y = np.array(
        [
            [1, 0],  # Action
            [0, 1],  # Romance
            [1, 0],  # Action
        ],
        dtype=int,
    )

    model.fit(texts, Y)

    # Use decision_function (LinearSVC) to ensure it runs
    scores = model.decision_function(["An epic romantic adventure"])
    assert scores.shape[1] == Y.shape[1]  # 2 genres
