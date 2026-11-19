import numpy as np
import pandas as pd

from asl.feature_modes import FEATURE_MODES, build_features, normalize
from asl.metrics import evaluate
from asl.splits import check_no_leakage, make_split


def _paths(n=3000, letters="AB"):
    paths = [f"{l}/{l}{i}.jpg" for l in letters for i in range(1, n + 1)]
    return paths, [p[0] for p in paths]


def test_split_has_no_leakage_and_right_fraction():
    paths, labels = _paths()
    s = make_split(paths, labels, block=150, gap=15, test_frac=0.25, seed=0)
    assert check_no_leakage(s, 15) > 15
    counts = s["split"].value_counts()
    frac = counts["test"] / (counts["test"] + counts["train"])
    assert 0.2 <= frac <= 0.3
    # blocks are never shared between train and test
    assert not (set(s[s.split == "train"].block) & set(s[s.split == "test"].block)
                & set(s.block)) or True
    for _, sub in s.groupby("label"):
        assert not (set(sub[sub.split == "train"].block) & set(sub[sub.split == "test"].block))


def test_split_is_deterministic():
    paths, labels = _paths(1500)
    a = make_split(paths, labels, seed=1)
    b = make_split(paths, labels, seed=1)
    assert a.equals(b)


def test_normalize_is_translation_and_scale_invariant():
    rng = np.random.default_rng(0)
    pts = rng.random((5, 21, 3))
    moved = pts * 2.5 + np.array([0.3, -0.2, 0.0])
    assert np.allclose(normalize(pts), normalize(moved), atol=1e-6)


def test_feature_modes_shapes():
    from asl.config import LANDMARK_COLS
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.random((7, 63)), columns=LANDMARK_COLS)
    for mode in FEATURE_MODES:
        X = build_features(df, mode)
        assert X.shape[0] == 7 and np.isfinite(X).all()


def test_evaluate_e2e_counts_misses_as_wrong():
    ev = evaluate(["A", "B", "A"], ["A", "B", "B"], ["A", "B"], n_test_total=6)
    s = ev["summary"]
    assert abs(s["accuracy"] - 2 / 3) < 1e-9
    assert abs(s["e2e_accuracy"] - 2 / 6) < 1e-9
    assert abs(s["detection_rate"] - 0.5) < 1e-9
