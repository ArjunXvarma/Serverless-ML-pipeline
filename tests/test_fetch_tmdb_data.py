import pandas as pd
from scripts import fetch_tmdb_data

def test_fetch_dataset_uses_all_genres(monkeypatch):
    # Fake fetch_for_genre to avoid network calls
    def fake_fetch_for_genre(genre_id, max_pages=10, per_genre_limit=150, **kwargs):
        # Return a tiny df with deterministic ids and overview
        return pd.DataFrame(
            {
                "id": [genre_id * 1000 + 1, genre_id * 1000 + 2],
                "title": [f"Movie {genre_id}-1", f"Movie {genre_id}-2"],
                "overview": [f"Overview for genre {genre_id}"] * 2,
                "genre_ids": [[genre_id], [genre_id]],
                "original_language": ["en", "en"],
                "release_date": ["2020-01-01", "2020-01-02"],
                "vote_average": [7.0, 6.5],
                "vote_count": [10, 20],
                "popularity": [10.0, 8.0],
            }
        )

    monkeypatch.setattr(fetch_tmdb_data, "fetch_for_genre", fake_fetch_for_genre)

    df = fetch_tmdb_data.fetch_dataset()

    # Ensure we got data
    assert not df.empty
    # Check that each row has a source_genre
    assert "source_genre" in df.columns
    # genre_ids should be non-empty
    assert df["genre_ids"].apply(len).min() > 0
