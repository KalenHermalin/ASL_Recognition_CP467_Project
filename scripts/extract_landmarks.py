"""Run preprocessing + MediaPipe over the dataset and cache landmarks to CSV. TODO (Kalen)."""


import argparse
from pathlib import Path

import pandas as pd

from asl.landmarks import HandLandmarker


def parse_args() -> argparse.Namespace:
    """Optional args: --variants (list of names, default all), --letters (default LETTERS),
    --limit (max images per letter, for quick subset runs), --overwrite."""

def list_images(letters: list[str], limit: int | None) -> list[tuple[Path, str]]:
    """Walk DATASET_DIR/<letter>/ and return (image_path, label) pairs, sorted."""

def process_variant(variant: str, images: list[tuple[Path, str]],
                    landmarker: HandLandmarker) -> pd.DataFrame:
    """For each image: imread, apply_variant, extract, result_to_row. tqdm progress.
    Time each stage (preprocess / mediapipe) and keep the totals."""

def save_csv(df: pd.DataFrame, variant: str) -> Path:
    """Reorder columns to CSV_COLUMNS and write LANDMARK_DIR/landmarks_<variant>.csv."""

def detection_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Detection rate overall and per letter."""

def main() -> None:
    """For each requested variant: skip if the CSV exists (unless --overwrite),
    otherwise process, save, and print the detection rate and the timing."""