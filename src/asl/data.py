"""Load cached landmark CSVs and attach the shared train/test split (Mustafa)."""
from pathlib import Path

import numpy as np
import pandas as pd

from asl.config import LANDMARK_COLS, LANDMARK_DIR, DATA_DIR
from asl.splits import (DEFAULT_SPLIT_PATH, SAMPLE_SPLIT_PATH, load_split, make_split,
                        save_split, check_no_leakage)

SAMPLE_DIR = DATA_DIR / "samples"


def run_tag(variant: str, lighting: str = "normal") -> str:
    return variant if lighting == "normal" else f"{variant}_{lighting}"


def landmark_csv_path(variant: str, lighting: str = "normal", sample: bool = False,
                      directory=None) -> Path:
    if sample:
        return SAMPLE_DIR / f"landmarks_{run_tag(variant, lighting)}_sample.csv"
    return Path(directory or LANDMARK_DIR) / f"landmarks_{run_tag(variant, lighting)}.csv"


def load_landmarks(variant: str, lighting: str = "normal", sample: bool = False,
                   directory=None) -> pd.DataFrame:
    """All rows of one landmark CSV (including detected == 0 rows)."""
    path = landmark_csv_path(variant, lighting, sample, directory)
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    df["detected"] = df["detected"].astype(int)
    return df


def get_split(df: pd.DataFrame, sample: bool = False, path=None) -> pd.DataFrame:
    """Load the saved split, or build + save it from `df` if it does not exist yet."""
    path = Path(path) if path else (SAMPLE_SPLIT_PATH if sample else DEFAULT_SPLIT_PATH)
    if path.exists():
        return load_split(path)
    split = make_split(df["image_path"], df["label"])
    check_no_leakage(split)
    save_split(split, path)
    return split


def attach_split(df: pd.DataFrame, split: pd.DataFrame) -> pd.DataFrame:
    """Add a `split` column (train / test / gap) and a `block` column by image_path.
    Rows not present in the split file are dropped."""
    out = df.merge(split[["image_path", "split", "block"]], on="image_path", how="inner")
    if len(out) < len(df):
        print(f"[data] {len(df) - len(out)} rows not in split file - dropped")
    return out


def train_test(df: pd.DataFrame):
    """(train detected-only, test all rows incl. undetected). Test keeps misses so the
    detection rate and end-to-end accuracy can be reported."""
    train = df[(df["split"] == "train") & (df["detected"] == 1)]
    test = df[df["split"] == "test"]
    return train, test


def landmark_array(df: pd.DataFrame) -> np.ndarray:
    """(n, 63) raw landmark matrix in MediaPipe order x0,y0,z0..x20,y20,z20."""
    return df[LANDMARK_COLS].to_numpy(dtype=float)
