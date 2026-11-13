"""Run one image through every preprocessing variant + MediaPipe and save the results.

    - Author: Kalen Hermalin
    - ID: 169048742
Usage:
    python scripts/try_image.py path/to/photo.jpg [--out results/figures]
"""
import argparse
from pathlib import Path

import cv2

from asl.config import PREPROCESS_VARIANTS, RESULTS_DIR, CSV_COLUMNS
from asl.landmarks import HandLandmarker, LandmarkResult, draw_landmarks, result_to_row
from asl.preprocessing import apply_variant, resize


def parse_args() -> argparse.Namespace:
    """Read the image path and the optional output folder from the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("image", type=Path, help="path to a photo to test")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "figures",
                        help="folder for the overlay images (default: results/figures)")
    return parser.parse_args()


def load_image(path: Path):
    """cv2.imread the file; raise a clear error if it can't be read (imread returns None)."""
    img = cv2.imread(str(path))
    if img is None:
        raise RuntimeError(f"Cannot read image: {path}")
    return img


def run_variant(img_bgr, variant: str, landmarker: HandLandmarker) -> tuple:
    """apply_variant -> extract. Return (processed_img, LandmarkResult)."""
    img = apply_variant(img_bgr, variant)
    res = landmarker.extract(img)
    return (img, res)


def save_overlay(img_bgr, result: LandmarkResult, variant: str, out_dir: Path) -> Path:
    """draw_landmarks onto the image and write <out_dir>/<variant>.png. Return the path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{variant}.png"
    if not cv2.imwrite(str(out_path), draw_landmarks(img_bgr, result)):
        raise RuntimeError(f"Could not write {out_path}")
    return out_path


def check_row(result: LandmarkResult, variant: str) -> None:
    """Build a row with result_to_row and assert its keys equal set(CSV_COLUMNS)."""
    row = result_to_row(result, "test.jpg", "?", variant)
    missing = set(CSV_COLUMNS) - set(row)
    extra = set(row) - set(CSV_COLUMNS)
    assert not missing and not extra, f"row/schema mismatch: missing={missing} extra={extra}"


def main() -> None:
    """Load the image, open one HandLandmarker, loop over PREPROCESS_VARIANTS:
    run_variant, check_row, save_overlay, and print one line per variant
    (name, detected, handedness)."""
    args = parse_args()
    img = load_image(args.image)
    img = resize(img, 1000)
    with HandLandmarker() as landmarker:
        for variant in PREPROCESS_VARIANTS:
            processed, result = run_variant(img, variant, landmarker)
            check_row(result, variant)
            out_path = save_overlay(processed, result, variant, args.out)
            print(f"{variant:12} detected={result.detected!s:5} "
                  f"handedness={result.handedness}  -> {out_path}")


if __name__ == "__main__":
    main()