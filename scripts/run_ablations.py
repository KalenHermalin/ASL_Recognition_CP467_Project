"""Stage-by-stage comparison: vary ONE stage at a time, everything else at the baseline config.

Stages (matching the proposal's Comparison Plan):
  preprocessing  every preprocessing variant that has a landmark CSV, tested on each lighting CSV found
  normalization  raw vs norm vs norm_rot
  features       norm (coords) vs eng (angles/distances) vs combined
  classifier     knn vs svm vs mlp
All runs use the same split file. Writes results/tables/ablation_<stage>.csv.

Usage: python scripts/run_ablations.py [--stages classifier features ...] [--sample]
Author: Mustafa Qasim
"""
import argparse

import pandas as pd

from asl.classifiers import CLASSIFIERS
from asl.config import LIGHTING_LEVELS, PREPROCESS_VARIANTS, RESULTS_DIR
from asl.data import landmark_csv_path
from asl.experiment import run_experiment

BASE = dict(variant="none", features="norm", classifier="svm")
STAGES = ["preprocessing", "normalization", "features", "classifier"]
SHOW = ["variant", "features", "classifier", "test_lighting", "detection_rate", "accuracy",
        "e2e_accuracy", "macro_f1", "confusable_f1", "fit_s", "predict_ms"]


def configs(stage: str, sample: bool):
    if stage == "preprocessing":
        variants = [v for v in PREPROCESS_VARIANTS if landmark_csv_path(v, "normal", sample).exists()]
        return [{**BASE, "variant": v} for v in variants]
    if stage == "normalization":
        return [{**BASE, "features": f} for f in ("raw", "norm", "norm_rot")]
    if stage == "features":
        return [{**BASE, "features": f} for f in ("norm", "eng", "combined")]
    if stage == "classifier":
        return [{**BASE, "classifier": c} for c in CLASSIFIERS]
    raise ValueError(stage)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stages", nargs="+", default=STAGES, choices=STAGES)
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--tune", action="store_true")
    args = ap.parse_args()

    out_dir = RESULTS_DIR / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    for stage in args.stages:
        rows = []
        for cfg in configs(stage, args.sample):
            # train on normal lighting, test on every lighting that has a CSV (normal first)
            tests = [l for l in LIGHTING_LEVELS if landmark_csv_path(cfg["variant"], l, args.sample).exists()]
            rows += run_experiment(**cfg, test_lightings=tests, sample=args.sample,
                                   tune=args.tune, save=False)
        if not rows:
            print(f"[{stage}] nothing to run")
            continue
        table = pd.DataFrame(rows)
        suffix = "_sample" if args.sample else ""
        table.to_csv(out_dir / f"ablation_{stage}{suffix}.csv", index=False)
        print(f"\n=== {stage} ===")
        print(table[[c for c in SHOW if c in table]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
