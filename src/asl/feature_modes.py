"""Feature modes for the evaluation framework (Mustafa).

PLACEHOLDER for Pranav's normalization/feature stage: simple but working versions so the
classifier + ablation code can run now. When his code lands, replace the bodies of
`normalize` / `engineered` (or have `build_features` call his functions); the rest of the
evaluation code only depends on `build_features(df, mode) -> (n, d) array`.

Modes:
  raw       63 raw MediaPipe coordinates
  norm      63 coords, wrist at origin, scaled by palm size
  norm_rot  norm + rotated so the wrist->middle-MCP axis points up
  eng       engineered features only (joint angles, distances) from norm landmarks
  combined  norm coords + engineered features

No handedness mirroring: the Kaggle set shows one hand throughout and MediaPipe's Left/Right
label is unreliable there (see docs/SCHEMA.md). Webcam data will need the calibrated mirror.
"""
import numpy as np

from asl.data import landmark_array

FEATURE_MODES = ["raw", "norm", "norm_rot", "eng", "combined"]

WRIST, MIDDLE_MCP = 0, 9
# (base, mcp, pip/ip, tip) chains per finger, MediaPipe indices
FINGERS = {
    "thumb": (1, 2, 3, 4),
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}
TIPS = [4, 8, 12, 16, 20]
PALM = [0, 5, 9, 13, 17]


def _angle(a, b, c):
    """Angle at b (radians) between segments b->a and b->c. Inputs (n, 3)."""
    v1, v2 = a - b, c - b
    cos = np.einsum("ij,ij->i", v1, v2) / (
        np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-9)
    return np.arccos(np.clip(cos, -1.0, 1.0))


def normalize(pts: np.ndarray, rotate: bool = False) -> np.ndarray:
    """pts (n, 21, 3) -> wrist at origin, scaled by palm size (wrist -> middle MCP length in
    xy), optionally rotated in the image plane so that wrist->middle-MCP points up."""
    pts = pts - pts[:, WRIST:WRIST + 1, :]
    palm = np.linalg.norm(pts[:, MIDDLE_MCP, :2], axis=1)
    pts = pts / (palm[:, None, None] + 1e-9)
    if rotate:
        vx, vy = pts[:, MIDDLE_MCP, 0], pts[:, MIDDLE_MCP, 1]
        theta = np.arctan2(vx, -vy)             # angle from "up" (image y points down)
        c, s = np.cos(theta), np.sin(theta)
        x, y = pts[:, :, 0].copy(), pts[:, :, 1].copy()
        pts[:, :, 0] = c[:, None] * x - s[:, None] * y
        pts[:, :, 1] = s[:, None] * x + c[:, None] * y
    return pts


def engineered(pts: np.ndarray) -> np.ndarray:
    """Engineered features from normalized landmarks (n, 21, 3) -> (n, d).
    15 joint angles, 10 fingertip-fingertip distances, 5 fingertip-palm-centre distances,
    8 thumb-tip distances to the other fingers' MCP and PIP joints."""
    feats = []
    for base, mcp, pip, tip in FINGERS.values():
        feats.append(_angle(pts[:, WRIST], pts[:, base], pts[:, mcp]))
        feats.append(_angle(pts[:, base], pts[:, mcp], pts[:, pip]))
        feats.append(_angle(pts[:, mcp], pts[:, pip], pts[:, tip]))
    for i in range(len(TIPS)):
        for j in range(i + 1, len(TIPS)):
            feats.append(np.linalg.norm(pts[:, TIPS[i]] - pts[:, TIPS[j]], axis=1))
    centre = pts[:, PALM].mean(axis=1)
    for t in TIPS:
        feats.append(np.linalg.norm(pts[:, t] - centre, axis=1))
    for other in ("index", "middle", "ring", "pinky"):
        _, mcp, pip, _ = FINGERS[other]
        feats.append(np.linalg.norm(pts[:, 4] - pts[:, mcp], axis=1))
        feats.append(np.linalg.norm(pts[:, 4] - pts[:, pip], axis=1))
    return np.stack(feats, axis=1)


def build_features(df, mode: str = "norm") -> np.ndarray:
    """Feature matrix for the rows of `df` (detected rows only)."""
    if mode not in FEATURE_MODES:
        raise ValueError(f"unknown feature mode {mode!r}; choose from {FEATURE_MODES}")
    raw = landmark_array(df)
    if mode == "raw":
        return raw
    pts = raw.reshape(-1, 21, 3)
    rotate = mode == "norm_rot"
    npts = normalize(pts, rotate=rotate)
    if mode in ("norm", "norm_rot"):
        return npts.reshape(len(npts), -1)
    eng = engineered(normalize(pts, rotate=False))
    if mode == "eng":
        return eng
    return np.hstack([npts.reshape(len(npts), -1), eng])
