import cv2
import numpy as np


MASK_SIZE = 64
_HOG = cv2.HOGDescriptor(
    (MASK_SIZE, MASK_SIZE),  # window
    (16, 16),                # block
    (8, 8),                  # block stride
    (8, 8),                  # cell
    9,                       # orientation bins
)


def segment_skin(image_bgr: np.ndarray, min_area_ratio: float = 0.01) -> np.ndarray | None:
    """Return a cleaned binary hand mask, or None if no plausible component is found."""
    if image_bgr is None or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("Expected a three-channel BGR image")

    ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
    lower = np.array([0, 133, 77], dtype=np.uint8)
    upper = np.array([255, 173, 127], dtype=np.uint8)
    mask = cv2.inRange(ycrcb, lower, upper)

    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    image_area = image_bgr.shape[0] * image_bgr.shape[1]
    if cv2.contourArea(contour) < min_area_ratio * image_area:
        return None

    hand_mask = np.zeros(mask.shape, dtype=np.uint8)
    cv2.drawContours(hand_mask, [contour], -1, 255, thickness=cv2.FILLED)
    return hand_mask


def extract_features(image_bgr: np.ndarray) -> np.ndarray | None:
    """Extract a fixed-length HOG + contour descriptor, or None when segmentation fails."""
    mask = segment_skin(image_bgr)
    if mask is None:
        return None

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    if area <= 0:
        return None

    x, y, width, height = cv2.boundingRect(contour)
    if width <= 0 or height <= 0:
        return None

    crop = mask[y:y + height, x:x + width]
    scale = min(MASK_SIZE / width, MASK_SIZE / height)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    resized = cv2.resize(
        crop, (resized_width, resized_height), interpolation=cv2.INTER_NEAREST
    )

    canvas = np.zeros((MASK_SIZE, MASK_SIZE), dtype=np.uint8)
    top = (MASK_SIZE - resized_height) // 2
    left = (MASK_SIZE - resized_width) // 2
    canvas[top:top + resized_height, left:left + resized_width] = resized

    hog = _HOG.compute(canvas).reshape(-1)

    hu = cv2.HuMoments(cv2.moments(contour)).reshape(-1)
    hu = -np.sign(hu) * np.log10(np.maximum(np.abs(hu), 1e-30))

    hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
    if hull_area <= 0:
        return None

    image_area = image_bgr.shape[0] * image_bgr.shape[1]
    geometry = np.array([
        area / image_area,
        width / height,
        area / (width * height),
        area / hull_area,
    ], dtype=np.float32)

    features = np.concatenate((hog, hu.astype(np.float32), geometry))
    if not np.isfinite(features).all():
        raise FloatingPointError("Non-finite classical features produced")
    return features