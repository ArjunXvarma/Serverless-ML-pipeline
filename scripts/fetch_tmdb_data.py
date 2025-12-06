import os
import requests
import time
import pandas as pd
from dotenv import load_dotenv
from ml_pipeline.data_utils import TMDB_GENRES

load_dotenv()

TMDB_API_ACCESS_TOKEN = os.getenv("TMDB_API_ACCESS_TOKEN")
BASE_URL = "https://api.themoviedb.org/3"
RAW_PATH = "data/raw/movies.csv"

def fetch_for_genre(genre_id: int, max_pages: int=20, per_genre_limit: int=200, language: str="en-US", min_vote_count: int=10, sleep_seconds: float=0.25):
    """
    Fetch movies for a specific genre using TMDB /discover/movie.

    Parameters:
        genre_id (int): TMDB genre ID.
        max_pages (int): Maximum pages to fetch from API.
        per_genre_limit (int): Maximum items to keep for this genre.
        language (str): Language filter for results.
        min_vote_count (int): Filter out movies with almost no ratings.
        sleep_seconds (float): Sleep time between API calls.

    Returns:
        pd.DataFrame: Movies DataFrame for this genre.
    """

    if TMDB_API_ACCESS_TOKEN is None:
        raise RuntimeError("TMDB API access token not set")

    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {TMDB_API_ACCESS_TOKEN}",
    }

    all_movies = []

    for page in range(1, max_pages + 1):
        params = {
            "with_genres": genre_id,
            "page": page,
            "language": language,
            "sort_by": "popularity.desc",
            "vote_count.gte": min_vote_count,  # filter weak/noisy entries
            "include_adult": "false",
        }

        print(f"Fetching genre={genre_id} page={page}...")

        response = requests.get(
            url=f"{BASE_URL}/discover/movie",
            headers=headers,
            params=params,
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()

        results = data.get("results", [])
        if not results:
            print(f"No more results for genre {genre_id}, stopping early.")
            break

        for m in results:
            all_movies.append(
                {
                    "id": m.get("id"),
                    "title": m.get("title"),
                    "overview": m.get("overview"),
                    "genre_ids": m.get("genre_ids", []),
                    "original_language": m.get("original_language"),
                    "release_date": m.get("release_date"),
                    "vote_average": m.get("vote_average"),
                    "vote_count": m.get("vote_count"),
                    "popularity": m.get("popularity"),
                }
            )

        total_pages = data.get("total_pages")
        if total_pages is not None and page >= total_pages:
            print(f"Reached last available page ({total_pages}) for genre {genre_id}.")
            break

        if len(all_movies) >= per_genre_limit:
            print(f"Reached per-genre limit ({per_genre_limit}) for genre {genre_id}.")
            break

        # avoid rate limiting
        time.sleep(sleep_seconds)

    limited_movies = all_movies[:per_genre_limit]
    return pd.DataFrame(limited_movies)


def fetch_dataset():
    all_dfs = []

    PER_GENRE = 150  

    for gid, name in TMDB_GENRES.items():
        df_gen = fetch_for_genre(
            genre_id=gid,
            max_pages=10,
            per_genre_limit=PER_GENRE,
        )
        df_gen["source_genre"] = name
        all_dfs.append(df_gen)

    df = pd.concat(all_dfs, ignore_index=True).drop_duplicates(subset=["id"])
    df = df[df["overview"].notna()]
    df = df[df["original_language"].str.contains("en", na=False)]
    return df


def main():
    os.makedirs(os.path.dirname(RAW_PATH), exist_ok=True)

    print("Fetching balanced per-genre dataset...")
    df = fetch_dataset()

    # final cleaning
    df = df[df["overview"].notna()]
    df = df[df["original_language"].str.contains("en", na=False)]

    df.to_csv(RAW_PATH, index=False)
    print(f"Saved {len(df)} movies to {RAW_PATH}")

    print("\nDataset summary:")
    print(df["source_genre"].value_counts())
    print(f"Total unique movies: {df['id'].nunique()}")

if __name__ == "__main__":
    main()
