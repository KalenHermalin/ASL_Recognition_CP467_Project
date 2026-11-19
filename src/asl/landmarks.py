"""Stage 2: MediaPipe Hands landmark extraction.
    - Author: Kalen Hermalin
    - ID: 169048742
    """
from dataclasses import dataclass
import numpy as np
import cv2
import mediapipe as mp
from asl.config import LANDMARK_COLS, MODEL_PATH

@dataclass
class LandmarkResult:
    detected: bool
    handedness: str | None          # "Left" / "Right"
    handedness_score: float | None
    landmarks: np.ndarray | None    # shape (63,), order x0,y0,z0 ... x20,y20,z20


class HandLandmarker:
    """Owns the MediaPipe Hands object so it is built once and reused."""

    def __init__(self, static_image_mode: bool = True, max_num_hands: int = 1, min_detection_conf: float = 0.5) -> None: 
        opts = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode =  mp.tasks.vision.RunningMode.IMAGE if static_image_mode else mp.tasks.vision.RunningMode.VIDEO,
            min_hand_detection_confidence=min_detection_conf,
            num_hands= max_num_hands,

        )
        self._landmarker = mp.tasks.vision.HandLandmarker.create_from_options(opts)

    def extract(self, img_bgr: np.ndarray) -> LandmarkResult: 
        # BGR -> RGB,  build a LandmarkResult
        # no hand found -> LandmarkResult(False, None, None, None)
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result =  self._landmarker.detect(img)
        if not result.hand_landmarks:
            return LandmarkResult(False, None, None, None)
        else:
            hand = result.hand_landmarks[0]          # list of 21 points
            landmarks = np.array([[p.x, p.y, p.z] for p in hand], dtype=np.float32).flatten()
            return LandmarkResult(True, result.handedness[0][0].category_name, result.handedness[0][0].score, landmarks)


    def close(self) -> None: 
        # release the MediaPipe object
        self._landmarker.close()

    def __enter__(self): 
        return self
    def __exit__(self, *exc): 
        # lets you write: with HandLandmarker() as lm:
        self.close()

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),          # thumb
    (0,5),(5,6),(6,7),(7,8),          # index
    (5,9),(9,10),(10,11),(11,12),     # middle
    (9,13),(13,14),(14,15),(15,16),   # ring
    (13,17),(17,18),(18,19),(19,20),  # pinky
    (0,17),                           # palm edge
]
def draw_landmarks(img_bgr: np.ndarray, result: LandmarkResult) -> np.ndarray:
    """Return a copy of the image with the hand skeleton drawn on it."""
    out = img_bgr.copy()
    if not result.detected:
        return out

    h, w = out.shape[:2]
    pts = result.landmarks.reshape(21, 3)
    # x, y are normalized (0-1); scale to pixels. z is ignored for drawing.
    px = [(int(x * w), int(y * h)) for x, y, _ in pts]

    # Scale drawing sizes with the image so lines stay visible on both 200px dataset
    # images and multi-megapixel phone photos.
    thickness = max(1, round(max(h, w) / 250))
    radius = max(2, thickness * 2)

    for a, b in HAND_CONNECTIONS:
        cv2.line(out, px[a], px[b], (0, 255, 0), thickness)       # green bones
    for p in px:
        cv2.circle(out, p, radius, (0, 0, 255), -1)               # red joints

    if result.handedness is not None:
        cv2.putText(out, result.handedness, (px[0][0], px[0][1] + radius * 5),
                    cv2.FONT_HERSHEY_SIMPLEX, max(0.5, thickness / 3), (0, 255, 255), thickness)
    return out

def result_to_row(result: LandmarkResult, image_path: str, label: str, variant: str,
                  lighting: str = "normal") -> dict:
    """Flatten into one CSV row matching CSV_COLUMNS in config.py."""
    row = {
        "image_path": image_path,
        "label": label,
        "variant": variant,
        "lighting": lighting,
        "detected": int(result.detected),
        "handedness": result.handedness,
        "handedness_score": result.handedness_score,
    }
    if result.detected:
        row.update(zip(LANDMARK_COLS, result.landmarks.tolist()))
    else:
        row.update({col: None for col in LANDMARK_COLS})
    return row