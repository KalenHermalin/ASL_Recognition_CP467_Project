"""Shared paths and constants. Import from here so the whole group agrees."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"            # Kaggle download lands here
LANDMARK_DIR = DATA_DIR / "landmarks"  # cached CSVs, one per preprocessing variant
WEBCAM_DIR = DATA_DIR / "webcam"      # group webcam clips / frames (<person>/<letter>/...)
RESULTS_DIR = ROOT / "results"

KAGGLE_DATASET = "grassknoted/asl-alphabet"

# 24 static letters (J and Z need motion)
LETTERS = list("ABCDEFGHIKLMNOPQRSTUVWXY")

# Preprocessing variants compared in the proposal
PREPROCESS_VARIANTS = ["none", "gaussian", "median", "gamma", "clahe"]

N_LANDMARKS = 21
LANDMARK_COLS = [f"{axis}{i}" for i in range(N_LANDMARKS) for axis in "xyz"]

# Landmark CSV schema (raw, un-normalized - normalization is a later stage)
CSV_COLUMNS = [
    "image_path", "label", "variant", "detected",
    "handedness", "handedness_score",
] + LANDMARK_COLS
