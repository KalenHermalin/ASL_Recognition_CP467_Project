"""Train/test split for the Kaggle ASL images (Mustafa).

The Kaggle images are numbered frames from continuous capture, so neighbouring frames
(A100, A101) are near-duplicates and a random split leaks. We split by BLOCKS of consecutive
frame numbers within each letter, and drop ("gap") train frames that sit next to a test block.

The split is saved once (results/splits/split.csv, keyed by image_path) and reused by every
experiment AND by the classical baseline, so all methods see identical train/test images.
"""
import re
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from asl.config import RESULTS_DIR

SPLIT_DIR = RESULTS_DIR / "splits"
DEFAULT_SPLIT_PATH = SPLIT_DIR / "split.csv"
SAMPLE_SPLIT_PATH = SPLIT_DIR / "split_sample.csv"

DEFAULT_BLOCK = 150      # frames per block
DEFAULT_GAP = 15         # frames dropped from train on each side of a test block
DEFAULT_TEST_FRAC = 0.25


def frame_number(image_path: str) -> int:
    """'A/A123.jpg' -> 123."""
    m = re.search(r"(\d+)\.[A-Za-z]+$", str(image_path))
    if m is None:
        raise ValueError(f"no frame number in {image_path!r}")
    return int(m.group(1))


def make_split(image_paths, labels, block: int = DEFAULT_BLOCK, gap: int = DEFAULT_GAP,
               test_frac: float = DEFAULT_TEST_FRAC, seed: int = 0) -> pd.DataFrame:
    """Return DataFrame[image_path, label, frame, block, split], split in {train, test, gap}.

    Per letter: frames are grouped into blocks of `block` consecutive frame numbers, about
    `test_frac` of the blocks (chosen at random, seeded) become test, and train frames within
    `gap` frames of a test block are marked 'gap' (excluded from both sets).
    """
    df = pd.DataFrame({"image_path": list(image_paths), "label": list(labels)})
    df = df.drop_duplicates("image_path").reset_index(drop=True)
    df["frame"] = df["image_path"].map(frame_number)
    df["block"] = (df["frame"] // block).astype(int)
    df["split"] = "train"

    for label, idx in df.groupby("label").groups.items():
        sub = df.loc[idx]
        blocks = np.sort(sub["block"].unique())
        n_test = max(1, int(round(test_frac * len(blocks))))
        rng = np.random.default_rng(seed + zlib.crc32(str(label).encode()))
        test_blocks = set(rng.choice(blocks, size=n_test, replace=False).tolist())

        is_test = sub["block"].isin(test_blocks)
        df.loc[sub.index[is_test], "split"] = "test"

        if gap > 0:
            frame, blk = sub["frame"], sub["block"]
            near = pd.Series(False, index=sub.index)
            for t in test_blocks:
                # train frame just after test block t (block t+1), or just before it (block t-1)
                near |= (blk == t + 1) & (frame - (t + 1) * block < gap)
                near |= (blk == t - 1) & (t * block - frame <= gap)
            df.loc[sub.index[near & ~is_test], "split"] = "gap"
    return df


def check_no_leakage(split: pd.DataFrame, gap: int = DEFAULT_GAP) -> int:
    """Smallest frame distance between any train and test frame of the same letter.
    Raises if it is not greater than `gap` (for gap=0, the check is that blocks are disjoint)."""
    smallest = np.inf
    for _, sub in split.groupby("label"):
        tr = np.sort(sub.loc[sub["split"] == "train", "frame"].to_numpy())
        te = sub.loc[sub["split"] == "test", "frame"].to_numpy()
        if len(tr) == 0 or len(te) == 0:
            continue
        pos = np.searchsorted(tr, te)
        lo = np.where(pos > 0, tr[np.clip(pos - 1, 0, len(tr) - 1)], -10**9)
        hi = np.where(pos < len(tr), tr[np.clip(pos, 0, len(tr) - 1)], 10**9)
        smallest = min(smallest, int(np.min(np.minimum(np.abs(te - lo), np.abs(hi - te)))))
    if smallest <= gap and gap > 0:
        raise AssertionError(f"train/test frames only {smallest} apart (gap={gap})")
    return int(smallest)


def save_split(split: pd.DataFrame, path=DEFAULT_SPLIT_PATH):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    split.to_csv(path, index=False)
    return path


def load_split(path=DEFAULT_SPLIT_PATH) -> pd.DataFrame:
    return pd.read_csv(path)
