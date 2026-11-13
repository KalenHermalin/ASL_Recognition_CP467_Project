"""Download what the project needs that isn't in git (each step skips if already done):

  1. MediaPipe hand_landmarker.task model -> models/
  2. Kaggle ASL Alphabet dataset          -> data/raw/

The dataset needs a Kaggle account + API token: kaggle.com/settings -> Create New Token,
then save kaggle.json to ~/.kaggle/kaggle.json (or set KAGGLE_USERNAME / KAGGLE_KEY).

Run once per machine (after `pip install -r requirements.txt`):
    python scripts/download_dataset.py
"""
import os
import sys
import urllib.request

from asl.config import KAGGLE_DATASET, MODEL_PATH, MODEL_URL, RAW_DIR


def ensure_model() -> None:
    if MODEL_PATH.exists():
        print(f"Model already present: {MODEL_PATH}")
        return
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading MediaPipe model to {MODEL_PATH} ...")
    tmp = MODEL_PATH.with_suffix(".part")  # avoid leaving a half-downloaded file
    try:
        urllib.request.urlretrieve(MODEL_URL, tmp)
        tmp.rename(MODEL_PATH)
    except Exception as e:
        tmp.unlink(missing_ok=True)
        sys.exit(f"Model download failed: {e}\n"
                 f"Download hand_landmarker.task manually (see the Hand Landmarker docs) "
                 f"and save it to {MODEL_PATH}")
    print("Model downloaded.")


def ensure_dataset() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if any(p for p in RAW_DIR.iterdir() if p.name != ".gitkeep"):
        print(f"{RAW_DIR} already has data - skipping dataset download.")
        return
    os.environ["KAGGLEHUB_CACHE"] = str(RAW_DIR)  # must be set before importing kagglehub
    import kagglehub
    try:
        path = kagglehub.dataset_download(KAGGLE_DATASET)
    except Exception as e:
        sys.exit(f"Dataset download failed: {e}\n"
                 "Check your Kaggle credentials (~/.kaggle/kaggle.json).")
    print("Dataset at:", path)


if __name__ == "__main__":
    ensure_model()
    ensure_dataset()
