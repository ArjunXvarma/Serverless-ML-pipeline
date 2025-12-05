import os
from huggingface_hub import HfApi, HfFolder, create_repo, upload_file
from dotenv import load_dotenv

load_dotenv()

HF_REPO_ID = "arjun-varma/tmdb-genre-classifier"
MODEL_LOCAL_PATH = "artifacts/genre_model.joblib"
META_LOCAL_PATH = "artifacts/genre_meta.json"

def upload_to_hub():
    token = os.getenv("HF_TOKEN")
    
    if token is None:
        raise RuntimeError("HF_TOKEN environment variable is not set")
    
    HfFolder.save_token(token)
    api = HfApi()

    create_repo(
        repo_id=HF_REPO_ID,
        token=token,
        repo_type="model",
        private=False,
        exist_ok=True,
    )

    upload_file(
        path_or_fileobj=MODEL_LOCAL_PATH,
        path_in_repo="genre_model.joblib",
        repo_id=HF_REPO_ID,
        token=token,
        repo_type="model",
        commit_message="Update genre model",
    )

    upload_file(
        path_or_fileobj=META_LOCAL_PATH,
        path_in_repo="genre_meta.json",
        repo_id=HF_REPO_ID,
        token=token,
        repo_type="model",
        commit_message="Update metadata",
    )


if __name__ == "__main__":
    upload_to_hub()