"""Shared paths and constants. Import from here so the whole group agrees."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"            # Kaggle download lands here
LANDMARK_DIR = DATA_DIR / "landmarks"  # cached CSVs, one per preprocessing variant
WEBCAM_DIR = DATA_DIR / "webcam"      # group webcam clips / frames (<person>/<letter>/...)
RESULTS_DIR = ROOT / "results"

KAGGLE_DATASET = "grassknoted/asl-alphabet"

DATASET_DIR = RAW_DIR / "datasets/grassknoted/asl-alphabet/versions/1/asl_alphabet_train/asl_alphabet_train"
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
    # Kaggle images are 200x200: 128 = downscale control, 300/400 = upscale
    "resize_128":  {"type": "resize",   "params": {"width": 128}},
    "resize_300":  {"type": "resize",   "params": {"width": 300}},
    "resize_400":  {"type": "resize",   "params": {"width": 400}},
}

PREPROCESS_VARIANTS = list(VARIANTS)

# Synthetic lighting conditions (see asl/lighting.py). "normal" = images untouched.
# brightness = pixel multiplier; noise_sigma = Gaussian noise std (0-255 scale).
LIGHTING_LEVELS = {
    "normal":        None,
    "dark_mild":     {"brightness": 0.50, "noise_sigma": 3.0},
    "dark_moderate": {"brightness": 0.25, "noise_sigma": 5.0},
    "dark_severe":   {"brightness": 0.10, "noise_sigma": 6.0},
    # control: same dimming as dark_moderate but NO noise, to separate dimming from noise
    "dark_moderate_nonoise": {"brightness": 0.25, "noise_sigma": 0.0},
}

N_LANDMARKS = 21
LANDMARK_COLS = [f"{axis}{i}" for i in range(N_LANDMARKS) for axis in "xyz"]
MEDIAPIPE_MODEL = MODEL_PATH = ROOT / "models" / "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
# Landmark CSV schema (raw, un-normalized - normalization is a later stage)
CSV_COLUMNS = [
    "image_path", "label", "variant", "lighting", "detected",
    "handedness", "handedness_score",
] + LANDMARK_COLS
