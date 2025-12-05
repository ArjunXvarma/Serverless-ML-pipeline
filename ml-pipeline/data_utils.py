import ast
import pandas as pd
from sklearn.model_selection import train_test_split
from .config import TrainingConfig
import numpy as np

TMDB_GENRES = {
    28: "Action",
    12: "Adventure",
    16: "Animation",
    35: "Comedy",
    80: "Crime",
    99: "Documentary",
    18: "Drama",
    10751: "Family",
    14: "Fantasy",
    36: "History",
    27: "Horror",
    10402: "Music",
    9648: "Mystery",
    10749: "Romance",
    878: "Science Fiction",
    10770: "TV Movie",
    53: "Thriller",
    10752: "War",
    37: "Western",
}

ALL_GENRES = sorted(set(TMDB_GENRES.values()))

def load_raw_movies(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    if df["genre_ids"].dtype == object:
        df["genre_ids"] = df["genre_ids"].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )
    
    return df


def build_multi_label_matrix(df: pd.DataFrame):
    genre_name_list = ALL_GENRES
    name_to_idx = {g: i for i, g in enumerate(genre_name_list)}

    Y = np.zeros((len(df), len(genre_name_list)), dtype=int)

    for i, row in df.iterrows():
        ids = row['genre_ids'] or []

        for gid in ids:
            name = TMDB_GENRES.get(gid)
            if name is None:
                continue

            j = name_to_idx[name]
            Y[i, j] = 1

    return Y, genre_name_list


def make_train_test(df: pd.DataFrame, cfg: TrainingConfig):
    X = df["overview"].astype(str).tolist()
    Y, genre_names = build_multi_label_matrix(df)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
    )

    return X_train, X_test, Y_train, Y_test, genre_names

