import os
from huggingface_hub import hf_hub_download, HfApi, create_repo, upload_file
from dotenv import load_dotenv
import json

load_dotenv()

HF_REPO_ID = "arjun-varma/tmdb-genre-classifier"
MODEL_LOCAL_PATH = "artifacts/genre_model.joblib"
META_LOCAL_PATH = "artifacts/genre_meta.json"

def load_current_metrics():
    try:
        old_meta_path = hf_hub_download(HF_REPO_ID, "genre_meta.json", repo_type="model")
        with open(old_meta_path) as f:
            old_meta = json.load(f)
        return old_meta["metrics"]["f1_micro"]
    except Exception:
        return None
    

def load_new_metrics():
    with open(META_LOCAL_PATH) as f:
        new_meta = json.load(f)
    return new_meta["metrics"]["f1_micro"]


def upload_to_hub():
    token = os.getenv("HF_TOKEN")
    if token is None:
        raise RuntimeError("HF_TOKEN not set")

    api = HfApi()

    create_repo(repo_id=HF_REPO_ID, token=token, repo_type="model", private=False, exist_ok=True)

    old_score = load_current_metrics()
    new_score = load_new_metrics()

    if old_score is not None and new_score <= old_score:
        print(f"Model NOT uploaded: new f1_micro={new_score:.4f} <= old={old_score:.4f}")
        return
    else:
        print(f"Uploading new model: f1_micro improved {old_score} → {new_score}")

    upload_file(path_or_fileobj="artifacts/genre_model.joblib", path_in_repo="genre_model.joblib",
                repo_id=HF_REPO_ID, token=token)

    upload_file(path_or_fileobj="artifacts/genre_meta.json", path_in_repo="genre_meta.json",
                repo_id=HF_REPO_ID, token=token)

if __name__ == "__main__":
    upload_to_hub()