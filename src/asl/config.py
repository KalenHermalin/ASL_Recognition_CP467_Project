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
VARIANTS = {
    "none":        {"type": "none",     "params": {}},
    "gaussian_5":  {"type": "filter",   "params": {"method": "gaussian", "ksize": 5}},
    "median_3":    {"type": "filter",   "params": {"method": "median",   "ksize": 3}},
    "gamma_1_5":   {"type": "gamma",    "params": {"gamma": 1.5}},
    "gamma_2_5":   {"type": "gamma",    "params": {"gamma": 2.5}},
    "clahe_2":     {"type": "clahe",    "params": {"clip_limit": 2.0, "tile_grid": 8}},
    "clahe_4":     {"type": "clahe",    "params": {"clip_limit": 4.0, "tile_grid": 8}},
}

PREPROCESS_VARIANTS = list(VARIANTS)

N_LANDMARKS = 21
LANDMARK_COLS = [f"{axis}{i}" for i in range(N_LANDMARKS) for axis in "xyz"]

# Landmark CSV schema (raw, un-normalized - normalization is a later stage)
CSV_COLUMNS = [
    "image_path", "label", "variant", "detected",
    "handedness", "handedness_score",
] + LANDMARK_COLS
