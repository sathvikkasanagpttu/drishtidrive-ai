"""Facial geometry, region cropping, and mouth aspect ratio utilities.

Provides reusable geometric utilities for cropping rectangular face/mouth subregions
and estimating oral cavity openness (Mouth Aspect Ratio - MAR) without external dependencies.
"""
from __future__ import annotations

from typing import Optional, Tuple
import cv2
import numpy as np


def extract_crop_rect(
    frame: np.ndarray, cx: int, cy: int, bw: int, bh: int, width: int, height: int
) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
    """Crop rectangular region around center point with image boundary clamping."""
    x1, y1 = max(0, cx - bw // 2), max(0, cy - bh // 2)
    x2, y2 = min(width, cx + bw // 2), min(height, cy + bh // 2)
    if x2 - x1 < 10 or y2 - y1 < 10:
        return None, None
    return frame[y1:y2, x1:x2].copy(), (x1, y1, x2, y2)


def compute_crop_mar(crop: Optional[np.ndarray]) -> float:
    """Estimate mouth openness ratio (MAR) from oral cavity contrast depth."""
    if crop is None or crop.size == 0:
        return 0.14
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
    h_c, w_c = gray.shape
    inner = gray[int(h_c * 0.25) : int(h_c * 0.75), int(w_c * 0.20) : int(w_c * 0.80)]
    if inner.size == 0:
        return 0.14
    mean_val = float(np.mean(gray))
    if mean_val < 15.0:
        return 0.14
    cavity_threshold = max(20.0, mean_val * 0.40)
    dark_pixels = np.sum(inner < cavity_threshold)
    open_ratio = float(dark_pixels) / float(inner.size)
    mar = 0.14 + float(open_ratio * 0.60)
    return round(float(np.clip(mar, 0.14, 0.85)), 2)
