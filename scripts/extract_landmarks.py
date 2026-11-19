"""Run preprocessing + MediaPipe over the dataset and cache landmarks to CSV. 
   - Author: Kalen Hermalin
    - ID: 169048742
Usage:  python scripts/extact_landmarks.py  <--variants -- letters --limit --lighting --overwrite>
."""


import argparse
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import cv2
from asl.landmarks import HandLandmarker, LandmarkResult, result_to_row
from asl.config import (PREPROCESS_VARIANTS, LETTERS, DATASET_DIR, CSV_COLUMNS, LANDMARK_DIR,
                        RESULTS_DIR, LIGHTING_LEVELS)
from asl.lighting import darken
from asl.preprocessing import apply_variant
import time
import zlib

def parse_args() -> argparse.Namespace:
    """Optional args: --variants (list of names, default all), --letters (default LETTERS),
    --limit (max images per letter, for quick subset runs), --lighting (default normal),
    --overwrite."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+", default=PREPROCESS_VARIANTS, help=f"Varients are: {",".join(PREPROCESS_VARIANTS)}")
    parser.add_argument("--letters", nargs="+", default=LETTERS, help=f"Possible Letters are: {",".join(LETTERS)}")
    parser.add_argument("--limit", type=int, default=None, help="The max number of images per letter (default is all)")


    parser.add_argument("--lighting", nargs="+", default=["normal"], choices=list(LIGHTING_LEVELS),
                        help="lighting conditions to run (default: normal = images untouched)")
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
    

def run_tag(variant: str, lighting: str) -> str:
    """Name used in output files. Normal lighting keeps the plain variant name."""
    return variant if lighting == "normal" else f"{variant}_{lighting}"


def process_variant(variant: str, images: list[tuple[Path, str]],
                    landmarker: HandLandmarker,
                    lighting: str = "normal") -> tuple[pd.DataFrame, dict[str, float]]:
    """For each image: imread, apply_variant, extract, result_to_row. tqdm progress.
    Time each stage (preprocess / mediapipe) and keep the totals."""
    rows = []
    t_pre = 0.0  # total seconds spent in apply_variant
    t_mp = 0.0   # total seconds spent in landmarker.extract
    for path, label in tqdm(images, desc=run_tag(variant, lighting)):
        img = cv2.imread(str(path))
        rel = str(path.relative_to(DATASET_DIR))
        if img is not None:
            if lighting != "normal":
                # Simulate the camera: darken the raw image BEFORE preprocessing (not timed).
                # Seed from the image path so every variant sees the identical dark image.
                img = darken(img, seed=zlib.crc32(rel.encode()), **LIGHTING_LEVELS[lighting])
            start_apply = time.perf_counter()
            img = apply_variant(img, variant)
            end_apply = time.perf_counter()

            start_extract = time.perf_counter()
            res = landmarker.extract(img)
            end_extract = time.perf_counter()
            t_pre += end_apply - start_apply
            t_mp += end_extract - start_extract
            rows.append(result_to_row(res, rel, label, variant, lighting))
        else:
            # No image found, treat as undetected
            rows.append(result_to_row(LandmarkResult(False, None, None, None), rel, label, variant, lighting))
    return pd.DataFrame(rows), {"preprocess_s": t_pre, "mediapipe_s": t_mp}

            

    

def save_csv(df: pd.DataFrame, variant: str, lighting: str = "normal") -> Path:
    """Reorder columns to CSV_COLUMNS and write LANDMARK_DIR/landmarks_<variant>[_<lighting>].csv."""
    LANDMARK_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = LANDMARK_DIR / f"landmarks_{run_tag(variant, lighting)}.csv"
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


def save_results(variant: str, summary: pd.DataFrame, timings: dict[str, float],
                 n_images: int, letters: list[str], limit: int | None,
                 lighting: str = "normal") -> None:
    """Write results/tables/detection_<tag>.csv and timing_<tag>.csv."""
    tables_dir = RESULTS_DIR / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    tag = run_tag(variant, lighting)
    summary.to_csv(tables_dir / f"detection_{tag}.csv", index_label="label")

    n = max(n_images, 1)
    pre_ms = timings["preprocess_s"] / n * 1000
    mp_ms = timings["mediapipe_s"] / n * 1000
    timing = pd.DataFrame([{
        "variant": variant,
        "lighting": lighting,
        "n_images": n_images,
        "letters": " ".join(letters),
        "limit_per_letter": limit,
        "preprocess_ms": round(pre_ms, 3),
        "mediapipe_ms": round(mp_ms, 3),
        "total_ms": round(pre_ms + mp_ms, 3),
    }])
    timing.to_csv(tables_dir / f"timing_{tag}.csv", index=False)


def main() -> None:
    """For each requested lighting condition and variant: skip if the CSV exists
    (unless --overwrite), otherwise process, save, and print the detection rate and timing."""
    args = parse_args()
    images = list_images(args.letters, args.limit)
    print(f"{len(images)} images | variants: {', '.join(args.variants)} "
          f"| lighting: {', '.join(args.lighting)}")

    with HandLandmarker() as landmarker:
        for lighting in args.lighting:
            for variant in args.variants:
                tag = run_tag(variant, lighting)
                csv_path = LANDMARK_DIR / f"landmarks_{tag}.csv"
                if csv_path.exists() and not args.overwrite:
                    print(f"[{tag}] {csv_path.name} already exists - skipping (use --overwrite to redo)")
                    continue

                pds, timings = process_variant(variant, images, landmarker, lighting)
                saved = save_csv(pds, variant, lighting)

                n = max(len(pds), 1)
                summary = detection_summary(pds)
                save_results(variant, summary, timings, len(pds), args.letters, args.limit, lighting)

                print(f"\n=== {tag} -> {saved.name} ===")
                print(summary)
                print(f"preprocess: {timings['preprocess_s'] / n * 1000:.2f} ms/image, "
                      f"mediapipe: {timings['mediapipe_s'] / n * 1000:.2f} ms/image\n")


if __name__ == "__main__":
    main()