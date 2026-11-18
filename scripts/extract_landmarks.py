"""Run preprocessing + MediaPipe over the dataset and cache landmarks to CSV. TODO (Kalen)."""


import argparse
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import cv2
from asl.landmarks import HandLandmarker, LandmarkResult, result_to_row
from asl.config import PREPROCESS_VARIANTS, LETTERS, DATASET_DIR, CSV_COLUMNS, LANDMARK_DIR
from asl.preprocessing import apply_variant
import time

def parse_args() -> argparse.Namespace:
    """Optional args: --variants (list of names, default all), --letters (default LETTERS),
    --limit (max images per letter, for quick subset runs), --overwrite."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+", default=PREPROCESS_VARIANTS, help=f"Varients are: {",".join(PREPROCESS_VARIANTS)}")
    parser.add_argument("--letters", nargs="+", default=LETTERS, help=f"Possible Letters are: {",".join(LETTERS)}")
    parser.add_argument("--limit", type=int, default=None, help="The max number of images per letter (default is all)")


    parser.add_argument("--overwrite", action="store_true",
                        help="redo variants whose CSV already exists (default: skip them)")
    return parser.parse_args()

def list_images(letters: list[str], limit: int | None) -> list[tuple[Path, str]]:
    """Walk DATASET_DIR/<letter>/ and return (image_path, label) pairs, sorted."""
    output = []
    for letter in letters:
        letter_dir = Path(DATASET_DIR / letter.upper())
        if letter_dir.is_dir():
            images = sorted(letter_dir.glob("*.jpg"))
            if limit is not None:
                selected_images = images[::len(images) // limit][:limit]
                for select_image in selected_images:
                    output.append((select_image, letter.upper()))
            else:
                for image in images:
                    output.append((image, letter.upper()))
        else:
            raise FileNotFoundError(f"Directory for letter {letter} does not exist")
    return output
    

def process_variant(variant: str, images: list[tuple[Path, str]],
                    landmarker: HandLandmarker) -> tuple[pd.DataFrame, dict[str, float]]:
    """For each image: imread, apply_variant, extract, result_to_row. tqdm progress.
    Time each stage (preprocess / mediapipe) and keep the totals."""
    rows = []
    t_pre = 0.0  # total seconds spent in apply_variant
    t_mp = 0.0   # total seconds spent in landmarker.extract
    for path, label in tqdm(images, desc=variant):
        img = cv2.imread(str(path))
        rel = str(path.relative_to(DATASET_DIR))
        if img is not None:
            start_apply = time.perf_counter()
            img = apply_variant(img, variant)
            end_apply = time.perf_counter()

            start_extract = time.perf_counter()
            res = landmarker.extract(img)
            end_extract = time.perf_counter()
            t_pre += end_apply - start_apply
            t_mp += end_extract - start_extract
            rows.append(result_to_row(res, rel, label, variant))
        else:
            # No image found, treat as undetected
            rows.append(result_to_row(LandmarkResult(False, None, None, None), rel, label, variant))
    return pd.DataFrame(rows), {"preprocess_s": t_pre, "mediapipe_s": t_mp}

            

    

def save_csv(df: pd.DataFrame, variant: str) -> Path:
    """Reorder columns to CSV_COLUMNS and write LANDMARK_DIR/landmarks_<variant>.csv."""
    LANDMARK_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = LANDMARK_DIR / f"landmarks_{variant}.csv"
    df_column_order = df[CSV_COLUMNS]
    df_column_order.to_csv(csv_path, index=False)
    return csv_path

def detection_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Detection rate overall and per letter."""
    by_letter = df.groupby("label")["detected"].agg(["mean", "sum", "count"])
    by_letter.columns = ["rate", "detected", "total"]
    overall = pd.DataFrame(
        {"rate": [df["detected"].mean()], "detected": [df["detected"].sum()], "total": [len(df)]},
        index=["ALL"],
    )
    return pd.concat([by_letter, overall]).round({"rate": 4})


def main() -> None:
    """For each requested variant: skip if the CSV exists (unless --overwrite),
    otherwise process, save, and print the detection rate and the timing."""
    args = parse_args()
    images = list_images(args.letters, args.limit)
    print(f"{len(images)} images, variants: {', '.join(args.variants)}")

    with HandLandmarker() as landmarker:
        for variant in args.variants:
            csv_path = LANDMARK_DIR / f"landmarks_{variant}.csv"
            if csv_path.exists() and not args.overwrite:
                print(f"[{variant}] {csv_path.name} already exists - skipping (use --overwrite to redo)")
                continue

            pds, timings = process_variant(variant, images, landmarker)
            saved = save_csv(pds, variant)

            n = max(len(pds), 1)
            print(f"\n=== {variant} -> {saved.name} ===")
            print(detection_summary(pds))
            print(f"preprocess: {timings['preprocess_s'] / n * 1000:.2f} ms/image, "
                  f"mediapipe: {timings['mediapipe_s'] / n * 1000:.2f} ms/image\n")


if __name__ == "__main__":
    main()