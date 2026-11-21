import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from asl.classical_baseline import extract_features
from asl.config import DATASET_DIR, LETTERS, RESULTS_DIR
from asl.metrics import evaluate, plot_confusion, top_confused
from asl.splits import (
    DEFAULT_SPLIT_PATH,
    check_no_leakage,
    make_split,
    save_split,
)


def load_or_create_split(path: Path) -> pd.DataFrame:
    """Load the shared split, or create it directly from the raw image folders."""
    if path.exists():
        return pd.read_csv(path)

    image_paths = []
    labels = []

    for label in LETTERS:
        label_dir = DATASET_DIR / label
        if not label_dir.is_dir():
            raise FileNotFoundError(f"Missing dataset folder: {label_dir}")

        for image_path in sorted(label_dir.glob("*.jpg")):
            image_paths.append(image_path.relative_to(DATASET_DIR).as_posix())
            labels.append(label)

    if not image_paths:
        raise RuntimeError(f"No .jpg images found under {DATASET_DIR}")

    split = make_split(image_paths, labels)
    check_no_leakage(split)
    save_split(split, path)
    return split


def extract_rows(rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, int]:
    """Read images and extract features; count segmentation failures as misses."""
    features = []
    labels = []
    misses = 0

    for image_path, label in rows[["image_path", "label"]].itertuples(
        index=False, name=None
    ):
        full_path = DATASET_DIR / str(image_path)
        image = cv2.imread(str(full_path))
        if image is None:
            raise FileNotFoundError(f"Could not read image: {full_path}")

        descriptor = extract_features(image)
        if descriptor is None:
            misses += 1
            continue

        features.append(descriptor)
        labels.append(str(label))

    if features:
        X = np.vstack(features).astype(np.float32)
    else:
        X = np.empty((0, 0), dtype=np.float32)

    return X, np.asarray(labels), misses


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--letters",
        nargs="+",
        choices=LETTERS,
        default=list("AEMNST"),
        help="Gesture classes to evaluate; default is A E M N S T",
    )
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT_PATH)
    args = parser.parse_args()

    letters = list(dict.fromkeys(args.letters))
    split = load_or_create_split(args.split)

    required = {"image_path", "label", "split"}
    missing_columns = required - set(split.columns)
    if missing_columns:
        raise ValueError(f"Split is missing columns: {sorted(missing_columns)}")

    rows = split[
        split["split"].isin(["train", "test"])
        & split["label"].isin(letters)
    ].copy()
    train_rows = rows[rows["split"] == "train"]
    test_rows = rows[rows["split"] == "test"]

    for name, subset in (("train", train_rows), ("test", test_rows)):
        missing_labels = sorted(set(letters) - set(subset["label"].unique()))
        if missing_labels:
            raise ValueError(f"No {name} split rows for labels: {missing_labels}")

    X_train, y_train, train_misses = extract_rows(train_rows)
    X_test, y_test, test_misses = extract_rows(test_rows)

    if len(X_train) == 0:
        raise RuntimeError("No training images produced valid skin masks")

    missing_train_labels = sorted(set(letters) - set(y_train))
    if missing_train_labels:
        raise RuntimeError(
            f"No valid training masks for labels: {missing_train_labels}"
        )

    model = make_pipeline(
        StandardScaler(),
        LinearSVC(class_weight="balanced", max_iter=5000, random_state=0),
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test) if len(X_test) else np.asarray([])
    result = evaluate(
        y_test,
        predictions,
        labels=letters,
        n_test_total=len(test_rows),
    )

    tables = RESULTS_DIR / "tables"
    figures = RESULTS_DIR / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    summary = {
        "classifier": "linear_svc",
        "letters": " ".join(letters),
        "n_train_total": len(train_rows),
        "n_train_detected": len(y_train),
        "train_segmentation_misses": train_misses,
        "test_segmentation_misses": test_misses,
        **result["summary"],
    }

    pd.DataFrame([summary]).to_csv(
        tables / "classical_baseline_summary.csv", index=False
    )
    result["per_class"].to_csv(tables / "classical_baseline_per_class.csv")
    top_confused(result["cm"], letters).to_csv(
        tables / "classical_baseline_confused.csv", index=False
    )
    plot_confusion(
        result["cm"],
        letters,
        figures / "classical_baseline_confusion.png",
        title=f"Classical skin-mask baseline: {' '.join(letters)}",
    )

    print(pd.Series(summary).to_string())


if __name__ == "__main__":
    main()