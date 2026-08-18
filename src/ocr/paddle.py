from __future__ import annotations

import logging
import threading

import numpy as np

from src.schemas import BoundingBox

logger = logging.getLogger(__name__)

_engine = None
_engine_lock = threading.Lock()


def _get_engine():
    """Lazily construct a single shared PaddleOCR engine (model load is expensive).

    We already deskew/perspective-correct and CLAHE-normalize upstream, so the
    engine's own doc-orientation/unwarping/textline-orientation passes are
    disabled to avoid redundant, slower preprocessing.
    """
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                from paddleocr import PaddleOCR

                kwargs = dict(
                    lang="nl",
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                )
                try:
                    _engine = PaddleOCR(cpu_threads=1, **kwargs)
                except TypeError:
                    logger.warning("PaddleOCR build does not accept cpu_threads; running without it")
                    _engine = PaddleOCR(**kwargs)
    return _engine


class RawOCRLine:
    """One recognized text line: text, confidence, and its bounding box."""

    def __init__(self, text: str, confidence: float, bbox: BoundingBox):
        self.text = text
        self.confidence = confidence
        self.bbox = bbox


def _poly_to_bbox(poly: list[list[float]]) -> BoundingBox:
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    return BoundingBox(x=int(x0), y=int(y0), width=int(x1 - x0), height=int(y1 - y0))


def _rect_to_bbox(rect: list[float]) -> BoundingBox:
    x0, y0, x1, y1 = rect
    return BoundingBox(x=int(x0), y=int(y0), width=int(x1 - x0), height=int(y1 - y0))


def run_ocr(image: np.ndarray) -> list[RawOCRLine]:
    """Run PaddleOCR on an image (crop or full page) and return recognized lines.

    Uses the PaddleOCR 3.x `predict()` pipeline API, which returns one Result
    object per page; `result.json` holds a dict of parallel arrays
    (`rec_texts`, `rec_scores`, and either `rec_polys` or `rec_boxes`),
    optionally nested under a top-level "res" key depending on pipeline.
    """
    engine = _get_engine()
    try:
        results = engine.predict(image)
    except Exception:
        logger.exception("PaddleOCR inference failed")
        return []

    lines: list[RawOCRLine] = []
    if not results:
        return lines

    for page in results:
        data = getattr(page, "json", page)
        if isinstance(data, dict) and "res" in data:
            data = data["res"]

        texts = data.get("rec_texts") or []
        scores = data.get("rec_scores") or []
        polys = data.get("rec_polys") or data.get("dt_polys")
        boxes = data.get("rec_boxes")

        for i, text in enumerate(texts):
            confidence = float(scores[i]) if i < len(scores) else 0.0
            if polys is not None and i < len(polys):
                bbox = _poly_to_bbox(polys[i])
            elif boxes is not None and i < len(boxes):
                bbox = _rect_to_bbox(boxes[i])
            else:
                continue
            lines.append(RawOCRLine(text=text, confidence=confidence, bbox=bbox))

    return lines


def run_ocr_single_line(image: np.ndarray) -> RawOCRLine | None:
    """Run OCR on a small crop expected to contain a single line; return the
    highest-confidence result, or None if nothing was recognized."""
    lines = run_ocr(image)
    if not lines:
        return None
    return max(lines, key=lambda line: line.confidence)
