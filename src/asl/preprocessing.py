"""Stage 1: image preprocessing
    - Author: Kalen Hermalin
    - ID: 169048742
"""
import numpy as np
import cv2
from typing import Literal
from asl.config import VARIANTS

def resize(img: np.ndarray, width: int | None = None) -> np.ndarray: 
    if width is None:
        return img
    h, w = img.shape[:2]
    scale = width / w
    if scale == 1:
        return img
    new_size = (width, round(h * scale))  # cv2 wants (width, height)
    if scale < 1:
        return cv2.resize(img,new_size, interpolation=cv2.INTER_AREA)
    else:
       return  cv2.resize(img, new_size, interpolation=cv2.INTER_CUBIC)

def filter_blur(img: np.ndarray, ksize: int = 3, method: Literal["gaussian", "median"] = "gaussian")-> np.ndarray:
    if ksize <= 0 or ksize % 2 == 0:
        raise ValueError("ksize must be a positive odd integer")
    match method:
        case "gaussian":
            return cv2.GaussianBlur(img, (ksize, ksize), 0)
        case "median":
            return cv2.medianBlur(img, ksize)
        case _:
            raise ValueError(f"Invalid method got: {method}\t Expected: gaussian or median")

def gamma_correction(img: np.ndarray, gamma: float = 1) -> np.ndarray:
    if gamma <= 0:
        raise ValueError("gamma must be a positive float")
    if gamma == 1:
        return img
    lut = np.array([
        round((i / 255) ** (1 / gamma) * 255)
        for i in range (256)
    ]).astype(np.uint8)
    return cv2.LUT(img, lut)


def CLAHE(img: np.ndarray, clip_limit: float = 2.0, tile_grid: int = 8) -> np.ndarray:
    if clip_limit <= 0:
        raise ValueError("clip_limit must be positive")
    if tile_grid < 1:
        raise ValueError("tile_grid should be greater than or equal to 1")
    bright = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(bright)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    l_processed = clahe.apply(l)
    processed = cv2.merge([l_processed, a, b])

    return cv2.cvtColor(processed, cv2.COLOR_LAB2BGR)
    
def apply_variant(img: np.ndarray, name: str) -> np.ndarray:
    if name not in VARIANTS:
        raise ValueError(f"Unknown variant: {name}")

    cfg = VARIANTS[name]
    kind = cfg["type"]        # "gamma", "clahe", "filter" or "none"
    params = cfg["params"]    # e.g. {"gamma": 1.5}
    match kind:
        case "gamma": return gamma_correction(img, **params)
        case "clahe": return CLAHE(img, **params)
        case "filter": return filter_blur(img, **params)
        case "none": return img
        case _:
            raise ValueError("name is not a valid variant")