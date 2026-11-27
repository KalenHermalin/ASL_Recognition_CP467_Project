"""Qualitative figures for the report: where preprocessing helps and where it fails.
    - Author: Kalen Hermalin
    - ID: 169048742

Re-creates the exact image each run saw (same darkening seed, same variant), and draws the
landmarks stored in the A-E per-image CSVs, so nothing here re-runs MediaPipe.

  results/report/qualitative_normal.png : none / gamma_2_5 / clahe_2 on four kinds of case
  results/report/qualitative_dark.png   : none / gaussian_5 / gamma_2_5 across darkness levels

Run from the project root:  python scripts/make_qualitative_figures.py
"""
import sys
import zlib
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from asl.config import DATASET_DIR, LIGHTING_LEVELS  # noqa: E402
from asl.lighting import darken  # noqa: E402
from asl.preprocessing import apply_variant  # noqa: E402

LM = ROOT / "data" / "landmarks_subset_A-E"
OUT = ROOT / "results" / "report"
OUT.mkdir(parents=True, exist_ok=True)
SZ = 200
CONNECTIONS = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),
               (9,13),(13,14),(14,15),(15,16),(13,17),(17,18),(18,19),(19,20),(0,17)]
GREEN, RED = "#1a9850", "#d73027"


def tag(v, lt):
    return v if lt == "normal" else f"{v}_{lt}"


_cache = {}


def table(v, lt):
    key = (v, lt)
    if key not in _cache:
        _cache[key] = pd.read_csv(LM / f"landmarks_{tag(v, lt)}.csv").set_index("image_path")
    return _cache[key]


def render(rel, v, lt):
    """Return (RGB image at SZ x SZ, landmark array (21,2) in pixels or None)."""
    img = cv2.imread(str(DATASET_DIR / rel))
    if lt != "normal":
        img = darken(img, seed=zlib.crc32(rel.encode()), **LIGHTING_LEVELS[lt])
    img = apply_variant(img, v)
    img = cv2.resize(img, (SZ, SZ), interpolation=cv2.INTER_AREA)
    row = table(v, lt).loc[rel]
    pts = None
    if int(row["detected"]) == 1:
        xy = row[[f"{a}{i}" for i in range(21) for a in "xy"]].to_numpy(dtype=float).reshape(21, 2)
        pts = xy * SZ
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), pts


def draw(ax, img, pts):
    ax.imshow(img)
    ax.set_xticks([]); ax.set_yticks([])
    colour = GREEN if pts is not None else RED
    for s in ax.spines.values():
        s.set_edgecolor(colour); s.set_linewidth(3)
    if pts is not None:
        for a, b in CONNECTIONS:
            ax.plot([pts[a, 0], pts[b, 0]], [pts[a, 1], pts[b, 1]], color="#00ff66", lw=1.6)
        ax.scatter(pts[:, 0], pts[:, 1], s=10, color="#ff2222", zorder=3)
    # landmarks can fall slightly outside the image; keep the frame fixed to the image
    ax.set_xlim(-0.5, SZ - 0.5)
    ax.set_ylim(SZ - 0.5, -0.5)
    if pts is None:
        ax.text(SZ / 2, SZ - 10, "no hand found", color="white", ha="center", va="bottom", fontsize=8,
                bbox=dict(facecolor=RED, edgecolor="none", pad=1.5))


rng = np.random.default_rng(0)

# ---------------- figure 1: normal lighting ----------------
n, g, c = table("none", "normal"), table("gamma_2_5", "normal"), table("clahe_2", "normal")
det = pd.DataFrame({"n": n["detected"], "g": g["detected"], "c": c["detected"], "label": n["label"]})
cases = [
    ("Gamma recovers a miss", det[(det.n == 0) & (det.g == 1)]),
    ("CLAHE loses a hand", det[(det.n == 1) & (det.g == 1) & (det.c == 0)]),
    ("All three detect", det[(det.n == 1) & (det.g == 1) & (det.c == 1)]),
    ("All three miss", det[(det.n == 0) & (det.g == 0) & (det.c == 0)]),
]
picks = []
for title, sub in cases:
    rel = sub.index[rng.integers(len(sub))]
    picks.append((title, rel, det.loc[rel, "label"]))
variants = ["none", "gamma_2_5", "clahe_2"]
fig, axes = plt.subplots(len(variants), len(picks), figsize=(6.6, 5.2))
for j, (title, rel, letter) in enumerate(picks):
    for i, v in enumerate(variants):
        img, pts = render(rel, v, "normal")
        draw(axes[i, j], img, pts)
        if i == 0:
            axes[i, j].set_title(f"{title}\n(letter {letter})", fontsize=8.5)
        if j == 0:
            axes[i, j].set_ylabel(v, fontsize=9)
fig.tight_layout(w_pad=0.4, h_pad=0.4)
fig.savefig(OUT / "qualitative_normal.png", dpi=200)
plt.close(fig)
print("normal picks:", picks)

# ---------------- figure 2: darkness ----------------
levels = ["normal", "dark_mild", "dark_moderate", "dark_severe"]
level_names = {"normal": "normal", "dark_mild": "mild", "dark_moderate": "moderate", "dark_severe": "severe"}
rows = ["none", "gaussian_5", "gamma_2_5"]
d = pd.DataFrame({"label": table("none", "normal")["label"]})
for v in rows:
    for lt in levels:
        d[f"{v}|{lt}"] = table(v, lt)["detected"]
# an image that shows the story: none survives mild but not moderate, gaussian_5 survives moderate,
# gamma_2_5 does well at normal but is lost once noise is present
cand = d[(d["none|normal"] == 1) & (d["none|dark_mild"] == 1) & (d["none|dark_moderate"] == 0)
         & (d["gaussian_5|dark_moderate"] == 1) & (d["gamma_2_5|normal"] == 1)
         & (d["gamma_2_5|dark_moderate"] == 0)]
rel = cand.index[rng.integers(len(cand))]
print("dark pick:", rel, "from", len(cand), "candidates")
fig, axes = plt.subplots(len(rows), len(levels), figsize=(6.0, 4.6))
for i, v in enumerate(rows):
    for j, lt in enumerate(levels):
        img, pts = render(rel, v, lt)
        draw(axes[i, j], img, pts)
        if i == 0:
            axes[i, j].set_title(level_names[lt], fontsize=9)
        if j == 0:
            axes[i, j].set_ylabel(v, fontsize=9)
fig.tight_layout(w_pad=0.4, h_pad=0.4)
fig.savefig(OUT / "qualitative_dark.png", dpi=200)
plt.close(fig)
print("Wrote figures to", OUT)
