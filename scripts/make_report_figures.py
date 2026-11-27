"""Build report tables and figures from the saved detection/timing tables.
    - Author: Kalen Hermalin
    - ID: 169048742

Reads  results/tables_subset_A-E/  (A-E grid: 10 variants x 5 lighting levels)
       results/tables/             (full 24-letter runs: none, gamma_2_5)
       data/landmarks_subset_A-E/  (per-image CSVs, for paired tests)
Writes results/report/*.csv and *.png

Run from the project root:  python scripts/make_report_figures.py
"""
import sys
from math import comb
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from asl.config import LIGHTING_LEVELS, VARIANTS  # noqa: E402

SUBSET_TABLES = ROOT / "results" / "tables_subset_A-E"
FULL_TABLES = ROOT / "results" / "tables"
SUBSET_LM = ROOT / "data" / "landmarks_subset_A-E"
OUT = ROOT / "results" / "report"
OUT.mkdir(parents=True, exist_ok=True)

VARIANT_ORDER = list(VARIANTS)
LIGHT_ORDER = list(LIGHTING_LEVELS)
LIGHT_LABELS = {
    "normal": "normal",
    "dark_mild": "mild\n(x0.50, s3)",
    "dark_moderate": "moderate\n(x0.25, s5)",
    "dark_severe": "severe\n(x0.10, s6)",
    "dark_moderate_nonoise": "moderate\nno noise",
}


def tag(variant, lighting):
    return variant if lighting == "normal" else f"{variant}_{lighting}"


def overall_rate(path):
    df = pd.read_csv(path)
    return float(df.loc[df["label"] == "ALL", "rate"].iloc[0]) * 100


# ---------- 1. detection matrix (A-E subset) ----------
rows = []
for v in VARIANT_ORDER:
    row = {"variant": v}
    for lt in LIGHT_ORDER:
        p = SUBSET_TABLES / f"detection_{tag(v, lt)}.csv"
        row[lt] = round(overall_rate(p), 1) if p.exists() else np.nan
    rows.append(row)
matrix = pd.DataFrame(rows).set_index("variant")
matrix.to_csv(OUT / "detection_matrix.csv")
print("\nDetection rate (%), A-E subset, 1000 images per cell\n", matrix, "\n")

fig, ax = plt.subplots(figsize=(8.2, 5.2))
im = ax.imshow(matrix.values, cmap="Blues", vmin=0, vmax=100, aspect="auto")
ax.set_xticks(range(len(LIGHT_ORDER)))
ax.set_xticklabels([LIGHT_LABELS[l] for l in LIGHT_ORDER], fontsize=9)
ax.set_yticks(range(len(matrix)))
ax.set_yticklabels(matrix.index, fontsize=9)
for i in range(matrix.shape[0]):
    for j in range(matrix.shape[1]):
        val = matrix.values[i, j]
        if np.isnan(val):
            continue
        ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=9,
                color="white" if val > 55 else "#222222")
ax.set_title("MediaPipe hand detection rate (%) by preprocessing and lighting\n"
             "(letters A-E, 1000 images per cell)", fontsize=11)
fig.colorbar(im, ax=ax, label="detection rate (%)", fraction=0.04)
fig.tight_layout()
fig.savefig(OUT / "detection_heatmap.png", dpi=200)
plt.close(fig)

# ---------- 2. detection vs darkness (line chart) ----------
levels = ["normal", "dark_mild", "dark_moderate", "dark_severe"]
key = ["none", "gaussian_5", "median_3", "gamma_2_5", "clahe_2"]
fig, ax = plt.subplots(figsize=(7.2, 4.4))
styles = {"none": ("#333333", "o", "-"), "gaussian_5": ("#1b9e77", "s", "-"),
          "median_3": ("#7570b3", "^", "-"), "gamma_2_5": ("#d95f02", "D", "-"),
          "clahe_2": ("#e7298a", "v", "--")}
for v in key:
    c, m, ls = styles[v]
    ax.plot(range(len(levels)), [matrix.loc[v, l] for l in levels],
            color=c, marker=m, linestyle=ls, linewidth=2, markersize=6, label=v)
ax.set_xticks(range(len(levels)))
ax.set_xticklabels(["normal", "mild", "moderate", "severe"])
ax.set_xlabel("synthetic darkening level")
ax.set_ylabel("detection rate (%)")
ax.set_ylim(0, 100)
ax.grid(axis="y", alpha=0.3)
ax.legend(frameon=False, fontsize=9)
ax.set_title("Detection rate drops with darkness; preprocessing choice matters", fontsize=11)
fig.tight_layout()
fig.savefig(OUT / "detection_vs_darkness.png", dpi=200)
plt.close(fig)

# ---------- 3. timing table ----------
trows = []
for v in VARIANT_ORDER:
    for lt in ["normal"]:
        p = SUBSET_TABLES / f"timing_{tag(v, lt)}.csv"
        if p.exists():
            t = pd.read_csv(p).iloc[0]
            trows.append({"variant": v, "preprocess_ms": t["preprocess_ms"],
                          "mediapipe_ms": t["mediapipe_ms"], "total_ms": t["total_ms"]})
timing = pd.DataFrame(trows)
timing.to_csv(OUT / "timing_table.csv", index=False)
print("Timing per image (ms), normal lighting, A-E subset\n", timing, "\n")

# ---------- 4. per-letter detection, full runs ----------
per_letter = None
for v in ["none", "gamma_2_5"]:
    p = FULL_TABLES / f"detection_{v}.csv"
    if p.exists():
        d = pd.read_csv(p).set_index("label")["rate"].mul(100).round(1).rename(v)
        per_letter = d.to_frame() if per_letter is None else per_letter.join(d)
if per_letter is not None:
    per_letter.to_csv(OUT / "per_letter_full.csv")
    letters = [l for l in per_letter.index if l != "ALL"]
    x = np.arange(len(letters))
    fig, ax = plt.subplots(figsize=(10, 4.2))
    w = 0.4
    ax.bar(x - w / 2, per_letter.loc[letters, "none"], w, color="#9aa7b5", label="none")
    if "gamma_2_5" in per_letter:
        ax.bar(x + w / 2, per_letter.loc[letters, "gamma_2_5"], w, color="#d95f02", label="gamma_2_5")
    ax.set_xticks(x)
    ax.set_xticklabels(letters)
    ax.set_ylabel("detection rate (%)")
    ax.set_ylim(0, 112)
    ax.set_yticks(range(0, 101, 20))
    ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=False, ncol=2, loc="upper center")
    ax.set_title("Per-letter detection rate, all 24 letters (3000 images each)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "per_letter_detection.png", dpi=200)
    plt.close(fig)
    print("Per-letter (full runs)\n", per_letter, "\n")


# ---------- 5. paired comparison vs 'none' (normal lighting, A-E) ----------
def sign_test_p(gained, lost):
    n = gained + lost
    if n == 0:
        return 1.0
    k = min(gained, lost)
    p = 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(p, 1.0)


base_p = SUBSET_LM / "landmarks_none.csv"
if base_p.exists():
    base = pd.read_csv(base_p, usecols=["image_path", "detected"]).set_index("image_path")["detected"].astype(bool)
    prow = []
    for v in VARIANT_ORDER:
        if v == "none":
            continue
        p = SUBSET_LM / f"landmarks_{v}.csv"
        if not p.exists():
            continue
        cur = pd.read_csv(p, usecols=["image_path", "detected"]).set_index("image_path")["detected"].astype(bool)
        joined = pd.concat([base.rename("b"), cur.rename("c")], axis=1, join="inner")
        gained = int((~joined.b & joined.c).sum())
        lost = int((joined.b & ~joined.c).sum())
        prow.append({"variant": v, "n": len(joined), "gained": gained, "lost": lost,
                     "net": gained - lost, "exact_p": sign_test_p(gained, lost)})
    paired = pd.DataFrame(prow)
    paired.to_csv(OUT / "paired_vs_none_normal.csv", index=False)
    print("Paired vs none (normal lighting)\n", paired, "\n")
else:
    print("Skipped paired test: per-image CSVs not found at", SUBSET_LM)

print("Wrote files to", OUT)
