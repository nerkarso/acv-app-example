from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

MAX_DIMENSION = 3500


def load_image(path: str) -> np.ndarray | None:
    """Load an image from disk. Returns None if unreadable/corrupt."""
    image = cv2.imread(path)
    if image is None or image.size == 0:
        logger.error("Failed to read image at %s", path)
        return None
    return image


def validate_image(image: np.ndarray | None) -> bool:
    if image is None:
        return False
    if image.ndim not in (2, 3):
        return False
    h, w = image.shape[:2]
    return h > 50 and w > 50


def resize_if_oversized(image: np.ndarray, max_dimension: int = MAX_DIMENSION) -> np.ndarray:
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= max_dimension:
        return image
    scale = max_dimension / longest
    return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def denoise_and_blur(gray: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(gray, (5, 5), 0)


def apply_clahe(image: np.ndarray) -> np.ndarray:
    """Contrast Limited Adaptive Histogram Equalization on an image (BGR or gray)."""
    if image.ndim == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)
        merged = cv2.merge((l_channel, a_channel, b_channel))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)


def normalize_resolution(image: np.ndarray, target_width: int = 2000) -> np.ndarray:
    h, w = image.shape[:2]
    if w == target_width:
        return image
    scale = target_width / w
    return cv2.resize(image, (target_width, int(h * scale)), interpolation=cv2.INTER_AREA)
