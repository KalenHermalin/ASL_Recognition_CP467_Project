"""One experiment = (preprocessing variant, features, classifier) trained on one lighting
condition and tested on one or more (Mustafa)."""
import time

import pandas as pd

from asl.classifiers import fit_classifier
from asl.config import LETTERS, RESULTS_DIR
from asl.data import (attach_split, get_split, landmark_csv_path, load_landmarks, run_tag,
                      train_test)
from asl.feature_modes import build_features
from asl.metrics import evaluate, plot_confusion, top_confused

TABLES = RESULTS_DIR / "tables"
FIGURES = RESULTS_DIR / "figures"


def _stage_timing(variant: str, lighting: str) -> dict:
    """Preprocess + MediaPipe ms/image from Kalen's timing_<tag>.csv, if it exists."""
    path = TABLES / f"timing_{run_tag(variant, lighting)}.csv"
    if not path.exists():
        return {}
    row = pd.read_csv(path).iloc[0]
    return {"preprocess_ms": float(row["preprocess_ms"]), "mediapipe_ms": float(row["mediapipe_ms"])}


def run_experiment(variant: str = "none", features: str = "norm", classifier: str = "svm",
                   train_lighting: str = "normal", test_lightings=None, sample: bool = False,
                   split_path=None, tune: bool = False, save: bool = True) -> list[dict]:
    """Train on the train split of `train_lighting`, test on the test split of each lighting in
    `test_lightings` (default: same as training). Returns one summary dict per test lighting."""
    test_lightings = list(test_lightings or [train_lighting])

    df_tr = load_landmarks(variant, train_lighting, sample)
    split = get_split(df_tr, sample, split_path)
    df_tr = attach_split(df_tr, split)
    train, _ = train_test(df_tr)

    t0 = time.perf_counter()
    X_train = build_features(train, features)
    feat_ms_train = (time.perf_counter() - t0) / max(len(train), 1) * 1000

    groups = train["label"] + "_" + train["block"].astype(str)
    t0 = time.perf_counter()
    model, best = fit_classifier(classifier, X_train, train["label"].to_numpy(),
                                 groups=groups, tune=tune)
    fit_s = time.perf_counter() - t0

    results = []
    for light in test_lightings:
        if not landmark_csv_path(variant, light, sample).exists():
            print(f"[skip] no landmark CSV for {run_tag(variant, light)}")
            continue
        df_te = attach_split(load_landmarks(variant, light, sample), split)
        _, test = train_test(df_te)
        test_det = test[test["detected"] == 1]

        t0 = time.perf_counter()
        X_test = build_features(test_det, features)
        feat_ms = (time.perf_counter() - t0) / max(len(test_det), 1) * 1000
        t0 = time.perf_counter()
        pred = model.predict(X_test)
        pred_ms = (time.perf_counter() - t0) / max(len(test_det), 1) * 1000

        ev = evaluate(test_det["label"].to_numpy(), pred, LETTERS, n_test_total=len(test))
        row = {
            "variant": variant, "features": features, "classifier": classifier,
            "train_lighting": train_lighting, "test_lighting": light,
            "n_train": len(train), **ev["summary"],
            "fit_s": round(fit_s, 3), "feature_ms": round(feat_ms, 4),
            "predict_ms": round(pred_ms, 4), "best_params": str(best) if best else "",
        }
        stage = _stage_timing(variant, light)
        if stage:
            total = stage["preprocess_ms"] + stage["mediapipe_ms"] + feat_ms + pred_ms
            row.update(stage, total_ms=round(total, 3), fps_est=round(1000 / total, 1))
        results.append(row)

        if save:
            tag = f"{run_tag(variant, light)}__train-{train_lighting}__{features}__{classifier}"
            if sample:
                tag += "__sample"
            for d in (TABLES, FIGURES):
                d.mkdir(parents=True, exist_ok=True)
            ev["per_class"].round(4).to_csv(TABLES / f"perclass_{tag}.csv")
            top_confused(ev["cm"], LETTERS).to_csv(TABLES / f"confused_{tag}.csv", index=False)
            plot_confusion(ev["cm"], LETTERS, FIGURES / f"confusion_{tag}.png", title=tag)
    return results


def append_summary(rows: list[dict], path=None) -> None:
    """Append result rows to results/tables/eval_summary.csv."""
    if not rows:
        return
    path = path or TABLES / "eval_summary.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame(rows)
    if path.exists():
        new = pd.concat([pd.read_csv(path), new], ignore_index=True)
    new.to_csv(path, index=False)
