"""Summarize MediaPipe's Left/Right labels in a landmark CSV, to calibrate mirroring for webcam data.
    - Author: Kalen Hermalin
    - ID: 169048742

MediaPipe's label assumes a mirrored (selfie) image and is unreliable on the Kaggle set (about
95% Right / 5% Left although the hand looks the same throughout). The rule we need is not "which
physical hand is this" but "does this capture path label a hand differently from the Kaggle
data?". So: record known-hand clips into separate folders, e.g.
    data/webcam/lucas_right/A/*.jpg    data/webcam/lucas_left/A/*.jpg
run  python scripts/extract_landmarks.py --source webcam --variants none
then  python scripts/check_handedness.py --csv data/landmarks/landmarks_webcam_none.csv
Each top-level folder is one group. A group whose majority label differs from --reference
(Kaggle's majority label, Right) needs mirroring so that it matches the Kaggle hand.

Usage: python scripts/check_handedness.py [--csv PATH] [--reference Right] [--min-frames 20]
"""
import argparse
from pathlib import Path

import pandas as pd

from asl.config import LANDMARK_DIR

CONSISTENT = 0.90   # majority share needed to trust a group's label


def group_of(image_path: str) -> str:
    """'lucas_left/A/img1.jpg' -> 'lucas_left'; Kaggle-style 'A/A1.jpg' -> '(all)'."""
    parts = Path(image_path).parts
    return parts[0] if len(parts) >= 3 else "(all)"


def summarize(df: pd.DataFrame, reference: str = "Right", min_frames: int = 20) -> pd.DataFrame:
    det = df[df["detected"] == 1].copy()
    det["group"] = det["image_path"].map(group_of)
    rows = []
    for group, sub in det.groupby("group"):
        share = sub["handedness"].value_counts(normalize=True)
        majority, majority_share = share.index[0], float(share.iloc[0])
        if len(sub) < min_frames:
            verdict = f"too few detected frames (<{min_frames}); record more"
        elif majority_share < CONSISTENT:
            verdict = "label inconsistent: do not rely on it; use a fixed rule for this capture path"
        elif majority == reference:
            verdict = "same as Kaggle: no mirroring"
        else:
            verdict = "differs from Kaggle: mirror these hands"
        rows.append({"group": group, "detected": len(sub),
                     "pct_Left": round(100 * float((sub["handedness"] == "Left").mean()), 1),
                     "pct_Right": round(100 * float((sub["handedness"] == "Right").mean()), 1),
                     "mean_score": round(float(sub["handedness_score"].mean()), 3),
                     "verdict": verdict})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, default=LANDMARK_DIR / "landmarks_webcam_none.csv")
    ap.add_argument("--reference", choices=["Left", "Right"], default="Right",
                    help="majority label of the Kaggle data (default Right)")
    ap.add_argument("--min-frames", type=int, default=20)
    args = ap.parse_args()

    df = pd.read_csv(args.csv, usecols=["image_path", "detected", "handedness", "handedness_score"])
    table = summarize(df, args.reference, args.min_frames)
    if table.empty:
        print("no detected hands in", args.csv)
        return
    print(table.to_string(index=False))
    print(f"\nreference (Kaggle) majority label: {args.reference}. Put the result in docs/SCHEMA.md.")


if __name__ == "__main__":
    main()
