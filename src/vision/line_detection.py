"""Content-based detection of answer-line and header-field regions.

No layout logic here may assume a fixed column count/position or fixed
pixel coordinates. Column and row boundaries are derived per-photo from
whitespace gaps in the binarized image; header fields are located by
proximity to OCR'd label text rather than fixed offsets.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from src.schemas import BoundingBox, AnswerLineRegion, HeaderFieldRegion

logger = logging.getLogger(__name__)

MIN_GAP_WIDTH_RATIO = 0.02  # min whitespace-column width, as fraction of region width
MIN_ROW_HEIGHT_RATIO = 0.01  # min text-row height, as fraction of region height
ROW_PADDING = 4


def _binarize(gray: np.ndarray) -> np.ndarray:
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 15
    )


def _find_content_bands(profile: np.ndarray, min_gap: int, min_band: int) -> list[tuple[int, int]]:
    """Given a 1D ink-density profile, return (start, end) index ranges of content bands
    separated by runs of near-zero density at least `min_gap` long."""
    is_ink = profile > 0
    bands: list[tuple[int, int]] = []
    start = None
    gap_len = 0
    for i, ink in enumerate(is_ink):
        if ink:
            if start is None:
                start = i
            gap_len = 0
        else:
            if start is not None:
                gap_len += 1
                if gap_len >= min_gap:
                    end = i - gap_len
                    if end - start >= min_band:
                        bands.append((start, end))
                    start = None
                    gap_len = 0
    if start is not None:
        end = len(profile) - gap_len
        if end - start >= min_band:
            bands.append((start, end))
    return bands


def detect_answer_line_regions(
    image: np.ndarray, region_bbox: BoundingBox | None = None
) -> list[AnswerLineRegion]:
    """Segment the answer-body region into per-line crops.

    Auto-detects column boundaries via a vertical whitespace-gap scan (works
    for 1, 2, or N columns) then, within each column, auto-detects row bands
    via a horizontal whitespace-gap scan. Each resulting crop should contain
    one "number + letter" unit.
    """
    if region_bbox is not None:
        x, y, w, h = region_bbox.x, region_bbox.y, region_bbox.width, region_bbox.height
        region = image[y : y + h, x : x + w]
    else:
        x = y = 0
        region = image
        h, w = region.shape[:2]

    if region.size == 0:
        return []

    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if region.ndim == 3 else region
    binary = _binarize(gray)

    col_density = binary.sum(axis=0) // 255
    min_gap = max(3, int(w * MIN_GAP_WIDTH_RATIO))
    min_band = max(10, int(w * 0.03))
    columns = _find_content_bands(col_density, min_gap=min_gap, min_band=min_band)
    if not columns:
        columns = [(0, w)]

    regions: list[AnswerLineRegion] = []
    for col_start, col_end in columns:
        col_slice = binary[:, col_start:col_end]
        row_density = col_slice.sum(axis=1) // 255
        min_row_gap = max(3, int(h * 0.006))
        min_row_band = max(8, int(h * MIN_ROW_HEIGHT_RATIO))
        rows = _find_content_bands(row_density, min_gap=min_row_gap, min_band=min_row_band)

        for row_start, row_end in rows:
            top = max(0, row_start - ROW_PADDING)
            bottom = min(h, row_end + ROW_PADDING)
            bbox = BoundingBox(
                x=x + col_start,
                y=y + top,
                width=col_end - col_start,
                height=bottom - top,
            )
            regions.append(AnswerLineRegion(bbox=bbox))

    return regions


def crop_region(image: np.ndarray, bbox: BoundingBox) -> np.ndarray:
    return image[bbox.y : bbox.y + bbox.height, bbox.x : bbox.x + bbox.width]


def detect_header_field_regions(
    image: np.ndarray,
    ocr_text_boxes: list[tuple[str, BoundingBox]],
    label_keywords: dict[str, list[str]],
    value_width_ratio: float = 2.5,
) -> list[HeaderFieldRegion]:
    """Locate header fields by proximity to a printed label word found by OCR.

    `ocr_text_boxes` is the output of a full-page OCR pass: (text, bbox) pairs.
    `label_keywords` maps a logical field name (e.g. "student_number") to a
    list of label substrings to match against OCR'd text (case-insensitive).
    For each match, the value region is assumed to extend to the right of the
    label (typical for a form field), sized relative to the label's own box
    rather than a fixed pixel offset.
    """
    regions: list[HeaderFieldRegion] = []
    img_h, img_w = image.shape[:2]

    for field_name, keywords in label_keywords.items():
        best_match: tuple[str, BoundingBox] | None = None
        for text, bbox in ocr_text_boxes:
            normalized = text.strip().lower()
            if any(kw.lower() in normalized for kw in keywords):
                best_match = (text, bbox)
                break

        if best_match is None:
            continue

        _, label_bbox = best_match
        value_x = min(img_w - 1, label_bbox.x + label_bbox.width)
        value_width = min(img_w - value_x, int(label_bbox.width * value_width_ratio))
        value_bbox = BoundingBox(
            x=value_x,
            y=max(0, label_bbox.y - int(label_bbox.height * 0.2)),
            width=max(1, value_width),
            height=int(label_bbox.height * 1.4) or label_bbox.height,
        )
        regions.append(HeaderFieldRegion(bbox=value_bbox, field_name=field_name))

    return regions
