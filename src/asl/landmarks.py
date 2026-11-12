"""Stage 2: MediaPipe Hands landmark extraction. TODO (Kalen)."""
from dataclasses import dataclass
import numpy as np

@dataclass
class LandmarkResult:
    detected: bool
    handedness: str | None          # "Left" / "Right"
    handedness_score: float | None
    landmarks: np.ndarray | None    # shape (63,), order x0,y0,z0 ... x20,y20,z20


class HandLandmarker:
    """Owns the MediaPipe Hands object so it is built once and reused."""

    def __init__(self, static_image_mode: bool = True, max_num_hands: int = 1,
                 min_detection_confidence: float = 0.5) -> None: ...

    def extract(self, img_bgr: np.ndarray) -> LandmarkResult: ...
        # BGR -> RGB, hands.process, build a LandmarkResult
        # no hand found -> LandmarkResult(False, None, None, None)

    def close(self) -> None: ...
        # release the MediaPipe object

    def __enter__(self): ...
    def __exit__(self, *exc): ...
        # lets you write: with HandLandmarker() as lm:


def draw_landmarks(img_bgr: np.ndarray, result: LandmarkResult) -> np.ndarray:
    """Return a copy of the image with the hand skeleton drawn on it."""

def result_to_row(result: LandmarkResult, image_path: str, label: str, variant: str) -> dict:
    """Flatten into one CSV row matching CSV_COLUMNS in config.py."""