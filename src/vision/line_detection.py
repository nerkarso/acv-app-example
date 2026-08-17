"""Content-based detection of header-field regions.

Answer lines and header fields are both located from a single full-page
PaddleOCR pass (its trained text-line detector is far more robust to camera
noise and printed ruled lines than a hand-rolled pixel-density/contour
segmentation would be): answer lines are matched directly against the
"number + letter" pattern in `src.services.processing_service`, and header
fields are located here by proximity to OCR'd label text rather than fixed
offsets.
"""

from __future__ import annotations

import logging
import re

import numpy as np

from src.schemas import BoundingBox, HeaderFieldRegion

logger = logging.getLogger(__name__)


def crop_region(image: np.ndarray, bbox: BoundingBox) -> np.ndarray:
    return image[bbox.y : bbox.y + bbox.height, bbox.x : bbox.x + bbox.width]


def detect_header_field_regions(
    image: np.ndarray,
    ocr_text_boxes: list[tuple[str, float, BoundingBox]],
    label_keywords: dict[str, list[str]],
    value_width_ratio: float = 2.5,
) -> list[HeaderFieldRegion]:
    """Locate header fields by proximity to a printed label word found by OCR.

    `ocr_text_boxes` is the output of a full-page OCR pass: (text, confidence,
    bbox) triples. `label_keywords` maps a logical field name (e.g.
    "student_number") to a list of label substrings to match against OCR'd
    text (case-insensitive).
    """
    regions: list[HeaderFieldRegion] = []
    img_w = image.shape[1]

    for field_name, keywords in label_keywords.items():
        best_match: tuple[str, float, BoundingBox, re.Match[str]] | None = None
        for text, confidence, bbox in ocr_text_boxes:
            stripped = text.strip()
            normalized = stripped.lower()
            match = next(
                (m for kw in keywords if (m := re.search(rf"\b{re.escape(kw.lower())}\b", normalized))),
                None,
            )
            if match is not None:
                best_match = (stripped, confidence, bbox, match)
                break

        if best_match is None:
            continue

        text, confidence, label_bbox, match = best_match
        remainder = text[match.end() :].strip(" :._-\t")
        if remainder:
            regions.append(
                HeaderFieldRegion(
                    bbox=label_bbox,
                    field_name=field_name,
                    inline_text=remainder,
                    inline_confidence=confidence,
                )
            )
            continue

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
