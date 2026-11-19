"""Create the shared train/test split file (results/splits/split.csv).

Everyone (classifiers, classical baseline) must use this one file so all methods are
evaluated on identical train/test images. Splits by blocks of consecutive frame numbers per
letter, with a gap, because neighbouring Kaggle frames are near-duplicates.

Usage: python scripts/make_split.py [--source CSV | --sample] [--block 150 --gap 15
       --test-frac 0.25 --seed 0 --out PATH]
Author: Mustafa Qasim
"""
import argparse

import pandas as pd

from asl.data import landmark_csv_path
from asl.splits import (DEFAULT_BLOCK, DEFAULT_GAP, DEFAULT_SPLIT_PATH, DEFAULT_TEST_FRAC,
                        SAMPLE_SPLIT_PATH, check_no_leakage, make_split, save_split)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", help="landmark CSV to read image paths from "
                                     "(default: data/landmarks/landmarks_none.csv)")
    ap.add_argument("--sample", action="store_true", help="use data/samples CSV, write split_sample.csv")
    ap.add_argument("--block", type=int, default=DEFAULT_BLOCK)
    ap.add_argument("--gap", type=int, default=DEFAULT_GAP)
    ap.add_argument("--test-frac", type=float, default=DEFAULT_TEST_FRAC)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", help="output path")
    args = ap.parse_args()

    src = args.source or landmark_csv_path("none", "normal", sample=args.sample)
    df = pd.read_csv(src, usecols=["image_path", "label"])
    split = make_split(df["image_path"], df["label"], args.block, args.gap,
                       args.test_frac, args.seed)
    dist = check_no_leakage(split, args.gap)
    out = save_split(split, args.out or (SAMPLE_SPLIT_PATH if args.sample else DEFAULT_SPLIT_PATH))

    counts = split["split"].value_counts()
    print(f"{len(split)} images from {src}\n{counts.to_string()}")
    print(f"test fraction of (train+test): {counts.get('test', 0) / (counts.get('train', 0) + counts.get('test', 0)):.3f}")
    print(f"closest train/test frames of the same letter: {dist} apart (gap={args.gap})")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
