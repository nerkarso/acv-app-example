"""Allow-list + confidence validation for raw OCR output.

Nothing here ever forces an uncertain reading into A/B/C/D or a fabricated
question number -- validation failures are surfaced via `needs_escalation`
and the caller decides whether to escalate, review, or store UNKNOWN.
"""

from __future__ import annotations

import re

from src.config import settings
from src.ocr.paddle import RawOCRLine
from src.schemas import OCRAnswerResult, OCRFieldResult, VALID_ANSWERS

# e.g. "9. A", "9) B", "9 - C", "9D", "9:A", "9 C." -- number, separator
# noise, letter, optional trailing separator noise (students commonly end
# the letter with a period, e.g. "2 C.")
_ANSWER_PATTERN = re.compile(r"^\s*(\d{1,3})\s*[.\-:)\]]?\s*([A-Da-d])\s*[.\-:)\]]?\s*$")

# Format: SE/00YY/000
STUDENT_NUMBER_PATTERN = re.compile(r"^SE/\d{4}/\d{3}$", re.IGNORECASE)


def parse_answer_line(raw: RawOCRLine, num_questions: int) -> OCRAnswerResult:
    """Parse a 'number + letter' pattern out of one OCR'd answer-line."""
    match = _ANSWER_PATTERN.match(raw.text)

    question_number: int | None = None
    detected_answer: str | None = None

    if match:
        parsed_number = int(match.group(1))
        parsed_letter = match.group(2).upper()
        if 1 <= parsed_number <= num_questions:
            question_number = parsed_number
        if parsed_letter in VALID_ANSWERS:
            detected_answer = parsed_letter

    needs_escalation = (
        raw.confidence < settings.paddle_ocr_confidence_threshold
        or question_number is None
        or detected_answer is None
    )

    return OCRAnswerResult(
        question_number=question_number,
        detected_answer=detected_answer,
        confidence=raw.confidence,
        raw_text=raw.text,
        needs_escalation=needs_escalation,
    )


def validate_header_field(
    raw: RawOCRLine | None,
    field_name: str,
    pattern: re.Pattern[str] | None = None,
) -> OCRFieldResult:
    """Validate a header field OCR result against an optional format regex.

    Empty output or low confidence always triggers escalation; when a
    pattern is supplied, a non-match also triggers escalation.
    """
    if raw is None or not raw.text.strip():
        return OCRFieldResult(
            field_name=field_name,
            raw_text="",
            parsed_value=None,
            confidence=0.0,
            needs_escalation=True,
        )

    cleaned = raw.text.strip()
    parsed_value: str | None = cleaned

    if pattern is not None and not pattern.match(cleaned):
        parsed_value = None

    needs_escalation = (
        raw.confidence < settings.paddle_ocr_confidence_threshold
        or parsed_value is None
    )

    return OCRFieldResult(
        field_name=field_name,
        raw_text=raw.text,
        parsed_value=parsed_value,
        confidence=raw.confidence,
        needs_escalation=needs_escalation,
    )


def confidence_tier(confidence: float) -> str:
    """Classify a confidence score into 'auto_accept' | 'escalate' | 'manual_review'."""
    if confidence >= settings.confidence_auto_accept:
        return "auto_accept"
    if confidence >= settings.confidence_escalate_min:
        return "escalate"
    return "manual_review"
