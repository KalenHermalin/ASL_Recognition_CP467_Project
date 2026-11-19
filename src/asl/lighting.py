"""Synthetic lighting conditions, used to test preprocessing in low light.

This simulates what a dim camera would give MediaPipe, so it is applied to the raw
image BEFORE any preprocessing variant. It is an approximation of real low light
(real footage also has colour casts and blur); webcam clips are the real test.
"""
import numpy as np


def darken(img: np.ndarray, brightness: float = 0.3, noise_sigma: float = 0.0,
           seed: int = 0) -> np.ndarray:
    """Scale pixel values by `brightness` (0-1] and add Gaussian noise (std in 0-255 units).

    The same (img, brightness, noise_sigma, seed) always gives the same output, so
    every preprocessing variant sees an identical darkened image. Returns uint8,
    same shape as the input.
    """
    if not 0 < brightness <= 1:
        raise ValueError("brightness must be in (0, 1]")
    if noise_sigma < 0:
        raise ValueError("noise_sigma must be >= 0")
    out = img.astype(np.float32) * brightness
    if noise_sigma > 0:
        rng = np.random.default_rng(seed)
        out += rng.normal(0.0, noise_sigma, img.shape).astype(np.float32)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)
