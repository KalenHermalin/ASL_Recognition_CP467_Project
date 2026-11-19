"""Train one classifier on landmark features and evaluate it on the held-out split.

Usage: python scripts/train_eval.py --variant none --features norm --classifier svm
         [--train-lighting normal] [--test-lighting normal dark_mild ...] [--tune] [--sample]
Writes results/tables/{eval_summary,perclass_*,confused_*}.csv and results/figures/confusion_*.png
Author: Mustafa Qasim
"""
import argparse

import pandas as pd

from asl.classifiers import CLASSIFIERS
from asl.config import LIGHTING_LEVELS, PREPROCESS_VARIANTS
from asl.experiment import append_summary, run_experiment
from asl.feature_modes import FEATURE_MODES


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", default="none", choices=PREPROCESS_VARIANTS)
    ap.add_argument("--features", default="norm", choices=FEATURE_MODES)
    ap.add_argument("--classifier", default="svm", choices=CLASSIFIERS)
    ap.add_argument("--train-lighting", default="normal", choices=list(LIGHTING_LEVELS))
    ap.add_argument("--test-lighting", nargs="+", choices=list(LIGHTING_LEVELS),
                    help="default: same as --train-lighting")
    ap.add_argument("--tune", action="store_true", help="grid search with group-aware CV on train")
    ap.add_argument("--sample", action="store_true", help="use data/samples CSVs")
    ap.add_argument("--split", help="split file (default results/splits/split[_sample].csv)")
    ap.add_argument("--no-summary", action="store_true", help="don't append to eval_summary.csv")
    args = ap.parse_args()

    rows = run_experiment(args.variant, args.features, args.classifier, args.train_lighting,
                          args.test_lighting, args.sample, args.split, args.tune)
    if not args.no_summary:
        append_summary(rows)
    cols = ["test_lighting", "detection_rate", "accuracy", "e2e_accuracy", "macro_f1",
            "confusable_f1", "fit_s", "predict_ms"]
    print(pd.DataFrame(rows)[cols].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
