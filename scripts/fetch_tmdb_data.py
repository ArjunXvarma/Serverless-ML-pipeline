import os
import requests
import time
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

TMDB_API_ACCESS_TOKEN = os.getenv('TMDB_API_ACCESS_TOKEN')
BASE_URL = "https://api.themoviedb.org/3"
RAW_PATH = "data/raw/movies.csv"

def fetch_tmdb_movies(pages=5, category='popular', language='en-US'):
    if TMDB_API_ACCESS_TOKEN is None:
        raise RuntimeError('API access token not set')
    
    for page in range(1, pages + 1):
        url = f'{BASE_URL}/movie/{category}'
        headers = {
            "accept": "application/json",
            "Authorization": f"Bearer {TMDB_API_ACCESS_TOKEN}"
        }

        response = requests.get(url=url, headers=headers)
        response.raise_for_status()
        data = response.json()

        movies = []
        for m in data.get('results', []):
            movies.append({
                'id': m.get('id'),
                "title": m.get("title"),
                "overview": m.get("overview"),
                "genre_ids": m.get("genre_ids", []),
                "original_language": m.get("original_language"),
                "release_date": m.get("release_date"),
                "vote_average": m.get("vote_average"),
                "vote_count": m.get("vote_count"),
                "popularity": m.get("popularity"),
            })

        return pd.DataFrame(movies)
    

def main():
    os.makedirs(os.path.dirname(RAW_PATH), exist_ok=True)

    df_pop = fetch_tmdb_movies(category='popular')
    df_top = fetch_tmdb_movies(category='top_rated')

    df = pd.concat([df_pop, df_top], ignore_index=True).drop_duplicates(subset=["id"])
    df = df[(df["original_language"] == "en") & df["overview"].notna()]
    df.to_csv(RAW_PATH, index=False)
    print(f"Saved {len(df)} movies to {RAW_PATH}")


if __name__ == '__main__':
    main()
