"""Evaluation metrics (Mustafa): accuracy, per-class P/R/F1, confusion matrix, end-to-end accuracy."""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

# Letters that differ mostly by thumb placement / fist shape
CONFUSABLE = list("AEMNST")


def evaluate(y_true, y_pred, labels, n_test_total: int | None = None) -> dict:
    """Metrics on the DETECTED test frames.

    n_test_total = number of test frames before dropping missed detections. If given:
      detection_rate   = len(y_true) / n_test_total
      e2e_accuracy     = correct / n_test_total   (a missed detection counts as wrong; this is
                         the number to compare against the classical baseline)
    Returns dict(summary=dict, per_class=DataFrame, cm=ndarray, labels=list).
    """
    labels = list(labels)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    p, r, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    per_class = pd.DataFrame({"label": labels, "precision": p, "recall": r, "f1": f1,
                              "support": sup}).set_index("label")
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    correct = int((y_true == y_pred).sum())

    present = per_class["support"] > 0
    cluster = per_class.loc[[l for l in CONFUSABLE if l in per_class.index and present[l]]]
    summary = {
        "accuracy": accuracy_score(y_true, y_pred) if len(y_true) else np.nan,
        "macro_f1": per_class.loc[present, "f1"].mean(),
        "confusable_f1": cluster["f1"].mean() if len(cluster) else np.nan,
        "n_test_detected": int(len(y_true)),
    }
    if n_test_total is not None:
        summary["n_test_total"] = int(n_test_total)
        summary["detection_rate"] = len(y_true) / n_test_total if n_test_total else np.nan
        summary["e2e_accuracy"] = correct / n_test_total if n_test_total else np.nan
    return {"summary": summary, "per_class": per_class, "cm": cm, "labels": labels}


def top_confused(cm: np.ndarray, labels, k: int = 10) -> pd.DataFrame:
    """Most frequent off-diagonal (true -> predicted) pairs."""
    rows = [(labels[i], labels[j], int(cm[i, j]))
            for i in range(len(labels)) for j in range(len(labels))
            if i != j and cm[i, j] > 0]
    out = pd.DataFrame(rows, columns=["true", "pred", "count"])
    return out.sort_values("count", ascending=False).head(k).reset_index(drop=True)


def plot_confusion(cm: np.ndarray, labels, path, title: str = "") -> None:
    """Row-normalised confusion matrix saved to `path`."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, label="row-normalised")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
