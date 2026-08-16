from __future__ import annotations

import logging

import cv2
import numpy as np

from src.schemas import DocumentDetectionResult, ErrorCode
from src.vision.preprocessing import denoise_and_blur

logger = logging.getLogger(__name__)

MIN_AREA_RATIO = 0.2  # candidate contour must cover at least this fraction of the image
MIN_ASPECT_RATIO = 0.5
MAX_ASPECT_RATIO = 2.0


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def detect_document_corners(image: np.ndarray) -> DocumentDetectionResult:
    """Find the largest valid quadrilateral in the image via contour detection."""
    h, w = image.shape[:2]
    image_area = h * w

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = denoise_and_blur(gray)
    edges = cv2.Canny(blurred, 50, 150)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return DocumentDetectionResult(success=False, error=ErrorCode.DOCUMENT_NOT_FOUND)

    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for contour in contours[:10]:
        area = cv2.contourArea(contour)
        if area < image_area * MIN_AREA_RATIO:
            continue

        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) != 4:
            continue

        pts = approx.reshape(4, 2).astype("float32")
        ordered = _order_corners(pts)

        width_top = np.linalg.norm(ordered[1] - ordered[0])
        width_bottom = np.linalg.norm(ordered[2] - ordered[3])
        height_left = np.linalg.norm(ordered[3] - ordered[0])
        height_right = np.linalg.norm(ordered[2] - ordered[1])

        avg_width = (width_top + width_bottom) / 2
        avg_height = (height_left + height_right) / 2
        if avg_width == 0 or avg_height == 0:
            continue

        aspect = avg_width / avg_height
        if not (MIN_ASPECT_RATIO <= aspect <= MAX_ASPECT_RATIO) and not (
            MIN_ASPECT_RATIO <= 1 / aspect <= MAX_ASPECT_RATIO
        ):
            continue

        return DocumentDetectionResult(
            success=True,
            corners=[tuple(pt) for pt in ordered.tolist()],
        )

    logger.warning("No valid document quadrilateral found among top contours")
    return DocumentDetectionResult(success=False, error=ErrorCode.DOCUMENT_NOT_FOUND)
