"""Run preprocessing + MediaPipe over the dataset and cache landmarks to CSV.
   - Author: Kalen Hermalin
    - ID: 169048742
Usage:  python scripts/extract_landmarks.py  <--variants --letters --limit --lighting --overwrite
                                              --only-split --split-file --source --webcam-dir>

Examples:
  python scripts/extract_landmarks.py --variants median_3 gaussian_5      # full Kaggle run, normal light
  python scripts/extract_landmarks.py --variants none --lighting dark_mild dark_moderate \\
         --only-split test                                                # dark runs on test images only
  python scripts/extract_landmarks.py --source webcam --variants none     # data/webcam/<person>/<letter>/
"""


import argparse
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import cv2
from asl.landmarks import HandLandmarker, LandmarkResult, result_to_row
from asl.config import (PREPROCESS_VARIANTS, LETTERS, DATASET_DIR, CSV_COLUMNS, LANDMARK_DIR,
                        RESULTS_DIR, LIGHTING_LEVELS, WEBCAM_DIR)
from asl.lighting import darken
from asl.preprocessing import apply_variant
from asl.splits import DEFAULT_SPLIT_PATH, load_split
import time
import zlib

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
WEBCAM_PREFIX = "webcam_"   # webcam CSVs are named landmarks_webcam_<variant>[_<lighting>].csv


def parse_args() -> argparse.Namespace:
    """Optional args: --variants (list of names, default all), --letters (default LETTERS),
    --limit (max images per letter, for quick subset runs), --lighting (default normal),
    --overwrite, --only-split (restrict to the train/test images of the shared split file),
    --split-file, --source (kaggle or webcam), --webcam-dir."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variants", nargs="+", default=PREPROCESS_VARIANTS,
                        help="Variants are: " + ",".join(PREPROCESS_VARIANTS))
    parser.add_argument("--letters", nargs="+", default=LETTERS,
                        help="Possible letters are: " + ",".join(LETTERS))
    parser.add_argument("--limit", type=int, default=None, help="The max number of images per letter (default is all)")


    parser.add_argument("--lighting", nargs="+", default=["normal"], choices=list(LIGHTING_LEVELS),
                        help="lighting conditions to run (default: normal = images untouched)")
    parser.add_argument("--overwrite", action="store_true",
                        help="redo variants whose CSV already exists (default: skip them)")
    parser.add_argument("--only-split", choices=["train", "test"], default=None,
                        help="only process the images the shared split file marks as train/test "
                             "(use test for dark runs: classifiers train on normal light only). "
                             "Dark lighting only.")
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT_PATH,
                        help="split file for --only-split (default: results/splits/split.csv, "
                             "create it with scripts/make_split.py)")
    parser.add_argument("--source", choices=["kaggle", "webcam"], default="kaggle",
                        help="kaggle = the downloaded dataset (default); webcam = "
                             "data/webcam/<person>/<letter>/*.jpg")
    parser.add_argument("--webcam-dir", type=Path, default=WEBCAM_DIR,
                        help="root folder for --source webcam")
    args = parser.parse_args()

    if args.only_split and "normal" in args.lighting:
        parser.error("--only-split is for dark runs; normal-lighting CSVs must contain every image "
                     "(the classifiers train on them)")
    if args.only_split and args.source == "webcam":
        parser.error("--only-split only applies to the Kaggle dataset (webcam has no split file)")
    if args.source == "webcam" and args.limit is not None:
        parser.error("--limit only applies to the Kaggle dataset")
    return args

def list_images(letters: list[str], limit: int | None, only_split: str | None = None,
                split_file: Path = DEFAULT_SPLIT_PATH) -> list[tuple[Path, str]]:
    """Walk DATASET_DIR/<letter>/ and return (image_path, label) pairs, sorted.
    If only_split is "train" or "test", keep only the images that the split file assigns to it
    (applied before --limit)."""
    keep = None
    if only_split is not None:
        if not Path(split_file).exists():
            raise FileNotFoundError(f"{split_file} not found - run scripts/make_split.py first")
        split = load_split(split_file)
        keep = set(split.loc[split["split"] == only_split, "image_path"])

    output = []
    for letter in letters:
        letter_dir = Path(DATASET_DIR / letter.upper())
        if letter_dir.is_dir():
            images = sorted(letter_dir.glob("*.jpg"))
            if keep is not None:
                images = [im for im in images if im.relative_to(DATASET_DIR).as_posix() in keep]
            if limit is not None and images:
                selected_images = images[::max(1, len(images) // limit)][:limit]
                for select_image in selected_images:
                    output.append((select_image, letter.upper()))
            else:
                for image in images:
                    output.append((image, letter.upper()))
        else:
            raise FileNotFoundError(f"Directory for letter {letter} does not exist")
    return output


def list_webcam_images(letters: list[str], webcam_dir: Path = WEBCAM_DIR) -> list[tuple[Path, str]]:
    """Webcam frames as (image_path, label) pairs, sorted. Layout: webcam_dir/<person>/<letter>/*.jpg
    (webcam_dir/<letter>/*.jpg also works). Folders starting with "_" or "." are ignored.
    The label is the letter folder name."""
    webcam_dir = Path(webcam_dir)
    if not webcam_dir.is_dir():
        raise FileNotFoundError(f"{webcam_dir} does not exist")
    output = []
    for letter in letters:
        letter = letter.upper()
        folders = [webcam_dir / letter]
        folders += [p / letter for p in sorted(webcam_dir.iterdir())
                    if p.is_dir() and not p.name.startswith(("_", "."))]
        for folder in folders:
            if not folder.is_dir():
                continue
            for image in sorted(folder.iterdir()):
                if image.suffix.lower() in IMAGE_SUFFIXES:
                    output.append((image, letter))
    if not output:
        raise FileNotFoundError(f"no images found under {webcam_dir}/<person>/<letter>/")
    return output


def run_tag(variant: str, lighting: str, prefix: str = "") -> str:
    """Name used in output files. Normal lighting keeps the plain variant name.
    prefix marks the source (empty for Kaggle, "webcam_" for webcam)."""
    tag = variant if lighting == "normal" else f"{variant}_{lighting}"
    return prefix + tag


def process_variant(variant: str, images: list[tuple[Path, str]],
                    landmarker: HandLandmarker,
                    lighting: str = "normal",
                    base_dir: Path = DATASET_DIR,
                    prefix: str = "") -> tuple[pd.DataFrame, dict[str, float]]:
    """For each image: imread, apply_variant, extract, result_to_row. tqdm progress.
    Time each stage (preprocess / mediapipe) and keep the totals. image_path in the rows is
    relative to base_dir."""
    rows = []
    t_pre = 0.0  # total seconds spent in apply_variant
    t_mp = 0.0   # total seconds spent in landmarker.extract
    for path, label in tqdm(images, desc=run_tag(variant, lighting, prefix)):
        img = cv2.imread(str(path))
        rel = path.relative_to(base_dir).as_posix()
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


def save_csv(df: pd.DataFrame, variant: str, lighting: str = "normal", prefix: str = "") -> Path:
    """Reorder columns to CSV_COLUMNS and write LANDMARK_DIR/landmarks_<variant>[_<lighting>].csv
    (with the source prefix for webcam runs)."""
    LANDMARK_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = LANDMARK_DIR / f"landmarks_{run_tag(variant, lighting, prefix)}.csv"
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
                 lighting: str = "normal", prefix: str = "", subset: str = "all") -> None:
    """Write results/tables/detection_<tag>.csv and timing_<tag>.csv.
    subset says which images the run covered ("all", or "test"/"train" for --only-split runs)."""
    tables_dir = RESULTS_DIR / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    tag = run_tag(variant, lighting, prefix)
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
        "subset": subset,
    }])
    timing.to_csv(tables_dir / f"timing_{tag}.csv", index=False)


def main() -> None:
    """For each requested lighting condition and variant: skip if the CSV exists
    (unless --overwrite), otherwise process, save, and print the detection rate and timing."""
    args = parse_args()
    if args.source == "webcam":
        base_dir, prefix, subset = args.webcam_dir, WEBCAM_PREFIX, "all"
        images = list_webcam_images(args.letters, base_dir)
    else:
        base_dir, prefix, subset = DATASET_DIR, "", args.only_split or "all"
        images = list_images(args.letters, args.limit, args.only_split, args.split_file)
    print(f"{len(images)} images ({args.source}, subset: {subset}) | variants: {', '.join(args.variants)} "
          f"| lighting: {', '.join(args.lighting)}")

    with HandLandmarker() as landmarker:
        for lighting in args.lighting:
            for variant in args.variants:
                tag = run_tag(variant, lighting, prefix)
                csv_path = LANDMARK_DIR / f"landmarks_{tag}.csv"
                if csv_path.exists() and not args.overwrite:
                    print(f"[{tag}] {csv_path.name} already exists - skipping (use --overwrite to redo)")
                    continue

                pds, timings = process_variant(variant, images, landmarker, lighting, base_dir, prefix)
                saved = save_csv(pds, variant, lighting, prefix)

                n = max(len(pds), 1)
                summary = detection_summary(pds)
                save_results(variant, summary, timings, len(pds), args.letters, args.limit, lighting,
                             prefix, subset)

                print(f"\n=== {tag} -> {saved.name} ===")
                print(summary)
                print(f"preprocess: {timings['preprocess_s'] / n * 1000:.2f} ms/image, "
                      f"mediapipe: {timings['mediapipe_s'] / n * 1000:.2f} ms/image\n")


if __name__ == "__main__":
    main()
