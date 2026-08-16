from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def warp_document(image: np.ndarray, corners: list[tuple[float, float]]) -> np.ndarray | None:
    """Apply a perspective transform to flatten the document to a top-down view.

    `corners` must be ordered top-left, top-right, bottom-right, bottom-left.
    Returns None if the transform cannot be computed (degenerate geometry).
    """
    try:
        src = np.array(corners, dtype="float32")
        (tl, tr, br, bl) = src

        width_top = np.linalg.norm(tr - tl)
        width_bottom = np.linalg.norm(br - bl)
        max_width = int(max(width_top, width_bottom))

        height_left = np.linalg.norm(bl - tl)
        height_right = np.linalg.norm(br - tr)
        max_height = int(max(height_left, height_right))

        if max_width < 10 or max_height < 10:
            logger.error("Degenerate document dimensions after corner detection")
            return None

        dst = np.array(
            [
                [0, 0],
                [max_width - 1, 0],
                [max_width - 1, max_height - 1],
                [0, max_height - 1],
            ],
            dtype="float32",
        )

        matrix = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(image, matrix, (max_width, max_height))
        return warped
    except cv2.error as exc:
        logger.error("Perspective transform failed: %s", exc)
        return None
