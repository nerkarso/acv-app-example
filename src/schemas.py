"""Pydantic models used to validate pipeline outputs before they hit the DB.

These are transport/validation objects for detection results and OCR
fields produced by the vision/OCR/grading layers -- distinct from the
SQLAlchemy models in database/models.py, which represent persisted rows.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, field_validator

VALID_ANSWERS = ("A", "B", "C", "D")


class AnswerState(str, Enum):
    CLEAR = "clear"
    STRUCK_THROUGH = "struck_through"
    BLANK = "blank"
    AMBIGUOUS = "ambiguous"


class DetectionMethod(str, Enum):
    OPENCV = "opencv"
    PADDLEOCR = "paddleocr"
    CLOUD_VLM = "cloud_vlm"
    LOCAL_VLM = "local_vlm"
    MANUAL = "manual"


class ReviewStatus(str, Enum):
    NOT_REQUIRED = "not_required"
    PENDING = "pending"
    CONFIRMED = "confirmed"
    CORRECTED = "corrected"


class SubmissionStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    NEEDS_REVIEW = "needs_review"
    FAILED = "failed"


class ErrorCode(str, Enum):
    INVALID_IMAGE = "INVALID_IMAGE"
    DOCUMENT_NOT_FOUND = "DOCUMENT_NOT_FOUND"
    DOCUMENT_TRANSFORM_FAILED = "DOCUMENT_TRANSFORM_FAILED"
    OCR_FAILED = "OCR_FAILED"
    STUDENT_NUMBER_UNREADABLE = "STUDENT_NUMBER_UNREADABLE"
    ANSWER_REGION_NOT_FOUND = "ANSWER_REGION_NOT_FOUND"
    ANSWER_LOW_CONFIDENCE = "ANSWER_LOW_CONFIDENCE"
    AI_PROVIDER_UNAVAILABLE = "AI_PROVIDER_UNAVAILABLE"
    DATABASE_ERROR = "DATABASE_ERROR"


class BoundingBox(BaseModel):
    x: int
    y: int
    width: int
    height: int


class DocumentDetectionResult(BaseModel):
    """Output of contour-based page detection + perspective transform."""

    success: bool
    corners: list[tuple[float, float]] | None = None
    error: ErrorCode | None = None
    processed_image_path: str | None = None


class LineRegion(BaseModel):
    """A single detected answer-line or header-field region, located by content."""

    bbox: BoundingBox
    crop_image_path: str | None = None


class AnswerLineRegion(LineRegion):
    """One detected number+letter unit in the answer body."""

    question_number_hint: int | None = None


class HeaderFieldRegion(LineRegion):
    field_name: str


class OCRFieldResult(BaseModel):
    """Result of running PaddleOCR on a header-field crop."""

    field_name: str
    raw_text: str
    parsed_value: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    needs_escalation: bool = False


class OCRAnswerResult(BaseModel):
    """Result of running PaddleOCR on a single answer-line crop."""

    question_number: int | None = None
    detected_answer: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    raw_text: str = ""
    needs_escalation: bool = False

    @field_validator("detected_answer")
    @classmethod
    def validate_answer_letter(cls, v: str | None) -> str | None:
        if v is not None and v not in VALID_ANSWERS:
            raise ValueError(f"detected_answer must be one of {VALID_ANSWERS} or None, got {v!r}")
        return v


class StrikeThroughResult(BaseModel):
    """Independent strike-through heuristic result for an answer-line crop."""

    is_struck_through: bool
    diagonal_density: float = 0.0


class AnswerDetection(BaseModel):
    """Common output shape for any detector (OCR, vision provider, or manual).

    This is the shape VisionProvider.classify_answer returns, reused as the
    canonical "one detected answer" shape across the pipeline.
    """

    answer: str | None = None  # 'A'|'B'|'C'|'D'|None -- never guessed
    state: AnswerState
    confidence: float = Field(ge=0.0, le=1.0)
    method: DetectionMethod

    @field_validator("answer")
    @classmethod
    def validate_answer_letter(cls, v: str | None) -> str | None:
        if v is not None and v not in VALID_ANSWERS:
            raise ValueError(f"answer must be one of {VALID_ANSWERS} or None, got {v!r}")
        return v


class HeaderExtractionResult(BaseModel):
    student_number: OCRFieldResult | None = None
    name: OCRFieldResult | None = None
    other_fields: dict[str, OCRFieldResult] = Field(default_factory=dict)


class GradedAnswer(BaseModel):
    question_number: int
    detected_answer: str | None
    correct_answer: str
    is_correct: bool | None  # None when not clear/gradable
    answer_state: AnswerState


class GradingResult(BaseModel):
    """Output of the deterministic grading engine."""

    submission_id: int | None = None
    graded_answers: list[GradedAnswer]
    correct_count: int
    incorrect_count: int
    unanswered_count: int
    score: float
    total_points: float
    percentage: float
    computed_at: datetime = Field(default_factory=datetime.utcnow)


class AnswerKeyRow(BaseModel):
    """One row parsed from CSV answer-key import."""

    question: int
    answer: str
    points: float = 1.0

    @field_validator("answer")
    @classmethod
    def validate_answer_letter(cls, v: str) -> str:
        v = v.strip().upper()
        if v not in VALID_ANSWERS:
            raise ValueError(f"answer must be one of {VALID_ANSWERS}, got {v!r}")
        return v
