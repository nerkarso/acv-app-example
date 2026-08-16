"""Strike-through heuristic: detect a diagonal ink stroke crossing a letter crop.

Independent of OCR -- this always overrides the OCR-detected letter when it
fires: the answer_state becomes 'struck_through' regardless of what letter
was read.
"""

from __future__ import annotations

import cv2
import numpy as np

from src.schemas import StrikeThroughResult

DIAGONAL_DENSITY_THRESHOLD = 0.35
HOUGH_MIN_LINE_LENGTH_RATIO = 0.5
DIAGONAL_ANGLE_TOLERANCE_DEG = 30.0


def detect_strike_through(crop: np.ndarray) -> StrikeThroughResult:
    """Check an answer-line crop for a diagonal high-density pixel run crossing
    the letter's bounding box."""
    if crop is None or crop.size == 0:
        return StrikeThroughResult(is_struck_through=False, diagonal_density=0.0)

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    h, w = gray.shape[:2]
    if h < 4 or w < 4:
        return StrikeThroughResult(is_struck_through=False, diagonal_density=0.0)

    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 10
    )

    min_line_length = max(5, int(min(h, w) * HOUGH_MIN_LINE_LENGTH_RATIO))
    lines = cv2.HoughLinesP(
        binary,
        rho=1,
        theta=np.pi / 180,
        threshold=15,
        minLineLength=min_line_length,
        maxLineGap=3,
    )

    if lines is None:
        return StrikeThroughResult(is_struck_through=False, diagonal_density=0.0)

    diagonal_pixels = 0
    total_line_pixels = 0
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = float(np.hypot(x2 - x1, y2 - y1))
        if length == 0:
            continue
        angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        angle = min(angle, 180 - angle)
        total_line_pixels += length
        if DIAGONAL_ANGLE_TOLERANCE_DEG <= angle <= (90 - DIAGONAL_ANGLE_TOLERANCE_DEG) or (
            90 + DIAGONAL_ANGLE_TOLERANCE_DEG <= angle
        ):
            diagonal_pixels += length

    diagonal_density = diagonal_pixels / max(1.0, float(w))
    is_struck = diagonal_density >= DIAGONAL_DENSITY_THRESHOLD * min(h, w)

    return StrikeThroughResult(
        is_struck_through=bool(is_struck),
        diagonal_density=float(diagonal_density),
    )
