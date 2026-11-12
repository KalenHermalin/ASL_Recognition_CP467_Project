"""Download the Kaggle ASL Alphabet dataset into data/raw/ (skips if present).

Needs Kaggle credentials: https://www.kaggle.com/settings -> Create New Token,
then save kaggle.json to ~/.kaggle/kaggle.json (or set KAGGLE_USERNAME / KAGGLE_KEY).
Run once per machine:  python scripts/download_dataset.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from asl.config import KAGGLE_DATASET, RAW_DIR  # noqa: E402

if any(p for p in RAW_DIR.iterdir() if p.name != ".gitkeep"):
    print(f"{RAW_DIR} already has data - skipping download.")
    sys.exit(0)

os.environ["KAGGLEHUB_CACHE"] = str(RAW_DIR)  # must be set before importing kagglehub
import kagglehub  # noqa: E402

path = kagglehub.dataset_download(KAGGLE_DATASET)
print("Dataset at:", path)
