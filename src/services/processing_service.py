"""Orchestrates the full per-image processing pipeline:

Image -> OpenCV -> content-based region detection -> PaddleOCR
  -> confidence check -> vision provider only when necessary
  -> manual review when necessary -> deterministic grading -> SQLite

A single failed paper must never stop a batch.
"""

from __future__ import annotations

import hashlib
import logging
import re
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

import cv2

from src.ai import get_vision_provider
from src.config import settings
from src.database.database import get_session
from src.database.repositories import (
    AnswerRepository,
    ExamRepository,
    ProcessingLogRepository,
    StudentRepository,
    SubmissionRepository,
)
from src.grading.grader import AnswerKeyEntry, DetectedAnswer, grade_submission
from src.ocr.paddle import RawOCRLine, run_ocr, run_ocr_single_line
from src.ocr.validation import STUDENT_NUMBER_PATTERN, parse_answer_line, validate_header_field
from src.schemas import (
    AnswerState,
    DetectionMethod,
    ErrorCode,
    OCRFieldResult,
    ReviewStatus,
    SubmissionStatus,
)
from src.vision.document_detection import detect_document_corners
from src.vision.line_detection import crop_region, detect_header_field_regions
from src.vision.perspective import warp_document
from src.vision.preprocessing import (
    apply_clahe,
    load_image,
    normalize_resolution,
    resize_if_oversized,
    validate_image,
)
from src.vision.strike_through import detect_strike_through

logger = logging.getLogger(__name__)

HEADER_LABEL_KEYWORDS = {
    "student_number": ["studentnummer", "student number", "student nr", "nummer", "no."],
    "name": ["naam", "name"],
    "subject": ["vak", "subject", "course"],
    "date": ["datum", "date"],
}


@dataclass
class SubmissionOutcome:
    submission_id: int
    file_name: str
    status: str
    error_code: str | None = None
    ocr_accepted_count: int = 0
    escalated_count: int = 0
    manual_review_count: int = 0


@dataclass
class BatchTally:
    processed: int = 0
    successful: int = 0
    needs_review: int = 0
    failed: int = 0
    ocr_accepted_count: int = 0
    escalated_count: int = 0
    outcomes: list[SubmissionOutcome] = field(default_factory=list)


def compute_file_hash(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()


def find_duplicate(exam_id: int, file_hash: str) -> int | None:
    with get_session() as session:
        repo = SubmissionRepository(session)
        existing = repo.find_by_hash(exam_id, file_hash)
        return existing.id if existing else None


def _save_upload(file_bytes: bytes, original_name: str) -> Path:
    ext = Path(original_name).suffix or ".jpg"
    dest = settings.uploads_path / f"{uuid.uuid4().hex}{ext}"
    dest.write_bytes(file_bytes)
    return dest


def _log(submission_id: int, stage: str, status: str, message: str | None = None, duration_ms: int | None = None) -> None:
    with get_session() as session:
        ProcessingLogRepository(session).log(submission_id, stage, status, message, duration_ms)


def _timed(fn: Callable[[], object]) -> tuple[object, int]:
    start = time.monotonic()
    result = fn()
    duration_ms = int((time.monotonic() - start) * 1000)
    return result, duration_ms


def create_pending_submission(exam_id: int, file_bytes: bytes, original_name: str) -> int:
    file_hash = compute_file_hash(file_bytes)
    saved_path = _save_upload(file_bytes, original_name)
    with get_session() as session:
        repo = SubmissionRepository(session)
        submission = repo.create(exam_id=exam_id, original_image_path=str(saved_path), file_hash=file_hash)
        return submission.id


def _extract_header_fields_ocr(full_page_lines: list[RawOCRLine], image) -> dict[str, OCRFieldResult]:
    """OCR-only header-field extraction. Escalation (if anything here comes
    back needing it) happens later, batched together with any uncertain
    answers into one combined per-submission vision call."""
    text_boxes = [(line.text, line.confidence, line.bbox) for line in full_page_lines]
    regions = detect_header_field_regions(image, text_boxes, HEADER_LABEL_KEYWORDS)

    fields: dict[str, OCRFieldResult] = {}
    for region in regions:
        if region.inline_text is not None:
            raw = RawOCRLine(text=region.inline_text, confidence=region.inline_confidence, bbox=region.bbox)
        else:
            crop = crop_region(image, region.bbox)
            raw = run_ocr_single_line(crop)
        pattern = STUDENT_NUMBER_PATTERN if region.field_name == "student_number" else None
        fields[region.field_name] = validate_header_field(raw, region.field_name, pattern)

    return fields


@dataclass
class _AnswerRow:
    question_number: int
    detected_answer: str | None
    confidence: float
    detection_method: DetectionMethod
    answer_state: AnswerState
    review_status: ReviewStatus
    crop_image_path: str | None
    needs_escalation: bool
    auto_accepted: bool = False


def _score_answers_ocr(
    full_page_lines: list[RawOCRLine], image, submission_id: int, num_questions: int
) -> dict[int, _AnswerRow]:
    """OCR-only answer scoring (strike-through + pattern parse + confidence
    gate). Escalation happens later, batched with header fields into one
    combined per-submission vision call.

    Answer-line candidates come from the full-page OCR pass's own text-line
    boxes (the same pass used for header fields) rather than a separate
    pixel-density/contour segmentation: PaddleOCR's trained line detector is
    far more robust to camera noise and printed ruled lines than a
    hand-rolled whitespace-gap or connected-component heuristic, and reusing
    it here avoids a second, redundant OCR pass per submission. Only lines
    that parse as a "number + letter" answer line are kept, one per question
    number (highest OCR confidence wins on a duplicate read).
    """
    candidates: dict[int, RawOCRLine] = {}
    for line in full_page_lines:
        result = parse_answer_line(line, num_questions)
        if result.question_number is None:
            continue
        existing = candidates.get(result.question_number)
        if existing is None or line.confidence > existing.confidence:
            candidates[result.question_number] = line

    rows: dict[int, _AnswerRow] = {}

    for question_number, raw in candidates.items():
        crop = crop_region(image, raw.bbox)
        crop_path = settings.crops_path / f"{submission_id}_{uuid.uuid4().hex}.png"
        try:
            cv2.imwrite(str(crop_path), crop)
            crop_path_str = str(crop_path)
        except Exception:
            crop_path_str = None

        strike = detect_strike_through(crop)
        ocr_result = parse_answer_line(raw, num_questions)
        confidence = ocr_result.confidence
        detected_answer = ocr_result.detected_answer
        answer_state = AnswerState.CLEAR
        review_status = ReviewStatus.NOT_REQUIRED
        needs_escalation = False
        auto_accepted = False

        if strike.is_struck_through:
            # A struck-through mark always needs a human decision
            answer_state = AnswerState.STRUCK_THROUGH
            review_status = ReviewStatus.PENDING
        elif detected_answer is None:
            answer_state = AnswerState.BLANK if not raw.text.strip() else AnswerState.AMBIGUOUS
            needs_escalation = True
        elif confidence < settings.confidence_auto_accept:
            needs_escalation = True
        else:
            auto_accepted = True

        rows[question_number] = _AnswerRow(
            question_number=question_number,
            detected_answer=detected_answer,
            confidence=confidence,
            detection_method=DetectionMethod.PADDLEOCR,
            answer_state=answer_state,
            review_status=review_status,
            crop_image_path=crop_path_str,
            needs_escalation=needs_escalation,
            auto_accepted=auto_accepted,
        )

    # Every question number gets a row
    for question_number in range(1, num_questions + 1):
        if question_number in rows:
            continue
        rows[question_number] = _AnswerRow(
            question_number=question_number,
            detected_answer=None,
            confidence=0.0,
            detection_method=DetectionMethod.PADDLEOCR,
            answer_state=AnswerState.BLANK,
            review_status=ReviewStatus.NOT_REQUIRED,
            crop_image_path=None,
            needs_escalation=True,
        )

    return rows


_USED_HEADER_FIELDS = {"student_number", "name"}


def _apply_escalation(
    rows: dict[int, _AnswerRow], fields: dict[str, OCRFieldResult], image, submission_id: int
) -> None:
    """One combined vision-provider call for every field/answer OCR left
    uncertain, merged back into `rows`/`fields` in place. Replaces one API
    call per uncertain item with at most one call per submission."""
    provider = get_vision_provider()

    # "subject"/"date" are located by HEADER_LABEL_KEYWORDS but never
    # persisted or used downstream
    field_names = [
        name
        for name, result in fields.items()
        if result.needs_escalation and name in _USED_HEADER_FIELDS
    ]
    question_numbers = [qn for qn, row in rows.items() if row.needs_escalation]

    if provider.is_available and (field_names or question_numbers):
        outcome = provider.read_submission(image, field_names, question_numbers)
        _log(
            submission_id,
            stage="escalation",
            status="escalated",
            message=(
                f"combined vision call: {len(field_names)} field(s) / {len(question_numbers)} "
                f"answer(s) requested; {len(outcome.fields)} field(s) / {len(outcome.answers)} "
                "answer(s) returned"
            ),
        )

        for name in field_names:
            escalated_field = outcome.fields.get(name)
            if escalated_field is None:
                continue
            parsed_value = escalated_field.parsed_value
            if parsed_value is not None and name == "student_number" and not STUDENT_NUMBER_PATTERN.match(
                parsed_value
            ):
                parsed_value = None
            fields[name] = escalated_field.model_copy(
                update={"parsed_value": parsed_value, "needs_escalation": parsed_value is None}
            )

        for qn in question_numbers:
            detection = outcome.answers.get(qn)
            if detection is None:
                continue
            row = rows[qn]
            row.detected_answer = detection.answer
            row.confidence = detection.confidence
            row.detection_method = DetectionMethod.CLOUD_VLM
            row.answer_state = detection.state
            row.needs_escalation = False
            # A struck-through mark always needs a human decision, same as
            # when strike-through is caught locally
            if detection.state == AnswerState.STRUCK_THROUGH or detection.answer is None:
                row.review_status = ReviewStatus.PENDING
            else:
                row.review_status = ReviewStatus.NOT_REQUIRED

    # Anything still unresolved (no provider configured, or the model's
    # response omitted it) falls back to manual review.
    for row in rows.values():
        if row.needs_escalation:
            row.review_status = ReviewStatus.PENDING
    for name, result in fields.items():
        if result.needs_escalation:
            _log(
                submission_id,
                stage="header_extraction",
                status="needs_review",
                message=f"{name}: low confidence or unparseable ('{result.raw_text}')",
            )


def _persist_answers(
    rows: dict[int, _AnswerRow], exam_id: int, submission_id: int
) -> tuple[list[DetectedAnswer], int, int, int]:
    """Returns (detected_answers, ocr_accepted_count, escalated_count, manual_review_count)."""
    with get_session() as session:
        exam_repo = ExamRepository(session)
        answer_repo = AnswerRepository(session)
        key_by_number = {q.question_number: q for q in exam_repo.get_answer_key(exam_id)}

        for question_number, row in sorted(rows.items()):
            key_entry = key_by_number.get(question_number)
            answer_repo.create(
                submission_id=submission_id,
                question_number=question_number,
                detected_answer=row.detected_answer,
                correct_answer=key_entry.correct_answer if key_entry else None,
                confidence=row.confidence,
                detection_method=row.detection_method.value,
                answer_state=row.answer_state.value,
                review_status=row.review_status.value,
                is_correct=None,
                question_id=key_entry.id if key_entry else None,
                crop_image_path=row.crop_image_path,
            )

    detected = [
        DetectedAnswer(
            question_number=row.question_number,
            detected_answer=row.detected_answer,
            answer_state=row.answer_state,
        )
        for _, row in sorted(rows.items())
    ]
    ocr_accepted = sum(1 for row in rows.values() if row.auto_accepted)
    escalated = sum(1 for row in rows.values() if row.detection_method == DetectionMethod.CLOUD_VLM)
    manual_review = sum(1 for row in rows.values() if row.review_status == ReviewStatus.PENDING)
    return detected, ocr_accepted, escalated, manual_review


def process_submission(submission_id: int, exam_id: int) -> SubmissionOutcome:
    with get_session() as session:
        SubmissionRepository(session).set_processing_started(submission_id)

    stale_files: list[str] = []
    with get_session() as session:
        submission = SubmissionRepository(session).get(submission_id)
        exam = ExamRepository(session).get(exam_id)
        if submission is None or exam is None:
            raise ValueError(f"Submission {submission_id} or exam {exam_id} not found")
        original_path = submission.original_image_path
        num_questions = exam.total_questions or 0

        # A reprocess (e.g. force-reprocess on the Upload page) must not
        # accumulate duplicate Answer rows, nor leak the previous attempt's
        # processed image / crop files on disk.
        if submission.processed_image_path:
            stale_files.append(submission.processed_image_path)
        for answer in AnswerRepository(session).list_for_submission(submission_id):
            if answer.crop_image_path:
                stale_files.append(answer.crop_image_path)
        AnswerRepository(session).delete_for_submission(submission_id)

    for path_str in stale_files:
        try:
            Path(path_str).unlink(missing_ok=True)
        except OSError:
            pass

    file_name = Path(original_path).name

    def fail(error_code: ErrorCode, message: str) -> SubmissionOutcome:
        with get_session() as session:
            SubmissionRepository(session).update_status(
                submission_id, SubmissionStatus.FAILED.value, error_code.value, message
            )
        _log(submission_id, stage="pipeline", status="failed", message=message)
        return SubmissionOutcome(submission_id=submission_id, file_name=file_name, status="failed", error_code=error_code.value)

    def needs_review(error_code: ErrorCode | None, message: str, image_to_use=None) -> None:
        with get_session() as session:
            SubmissionRepository(session).update_status(
                submission_id,
                SubmissionStatus.NEEDS_REVIEW.value,
                error_code.value if error_code else None,
                message,
            )
        _log(submission_id, stage="pipeline", status="needs_review", message=message)

    image = load_image(original_path)
    if image is None or not validate_image(image):
        return fail(ErrorCode.INVALID_IMAGE, "Image could not be read or is corrupt")

    image = resize_if_oversized(image)

    detection = detect_document_corners(image)
    document_ok = detection.success
    working_image = image
    document_error_code: ErrorCode | None = None
    document_error_message: str | None = None

    if not document_ok or detection.corners is None:
        document_error_code = ErrorCode.DOCUMENT_NOT_FOUND
        document_error_message = "Document contour not found; using original image"
        needs_review(document_error_code, document_error_message)
    else:
        warped = warp_document(image, detection.corners)
        if warped is None:
            document_ok = False
            document_error_code = ErrorCode.DOCUMENT_TRANSFORM_FAILED
            document_error_message = "Perspective transform failed; using original image"
            needs_review(document_error_code, document_error_message)
        else:
            working_image = warped

    working_image = normalize_resolution(working_image)
    working_image = apply_clahe(working_image)

    processed_path = settings.processed_path / f"{submission_id}_{uuid.uuid4().hex}.png"
    cv2.imwrite(str(processed_path), working_image)
    with get_session() as session:
        SubmissionRepository(session).set_processed_image_path(submission_id, str(processed_path))

    full_page_lines = run_ocr(working_image)

    header_fields = _extract_header_fields_ocr(full_page_lines, working_image)
    answer_rows = _score_answers_ocr(full_page_lines, working_image, submission_id, num_questions)
    _apply_escalation(answer_rows, header_fields, working_image, submission_id)

    student_number_field = header_fields.get("student_number")
    student_number = student_number_field.parsed_value if student_number_field else None
    name_field = header_fields.get("name")
    name = name_field.parsed_value if name_field else None

    student_unreadable = not student_number
    if student_unreadable:
        _log(submission_id, stage="header_extraction", status="needs_review", message="STUDENT_NUMBER_UNREADABLE")
    else:
        with get_session() as session:
            student_repo = StudentRepository(session)
            student = student_repo.get_or_create(student_number, name)
            SubmissionRepository(session).set_student(submission_id, student.id)

    detected_answers, ocr_accepted, escalated, manual_review_count = _persist_answers(
        answer_rows, exam_id, submission_id
    )

    with get_session() as session:
        exam_repo = ExamRepository(session)
        key_rows = exam_repo.get_answer_key(exam_id)
        answer_key = [
            AnswerKeyEntry(question_number=q.question_number, correct_answer=q.correct_answer, points=q.points)
            for q in key_rows
        ]

    result = grade_submission(detected_answers, answer_key, submission_id=submission_id)

    with get_session() as session:
        answer_repo = AnswerRepository(session)
        db_answers = {a.question_number: a for a in answer_repo.list_for_submission(submission_id)}
        for graded in result.graded_answers:
            answer = db_answers.get(graded.question_number)
            if answer is not None:
                answer.is_correct = graded.is_correct
        session.flush()

    final_status = SubmissionStatus.COMPLETED.value
    review_error_code = None
    review_message = None
    if not document_ok:
        final_status = SubmissionStatus.NEEDS_REVIEW.value
        # set_processing_completed always overwrites error_code/message
        review_error_code = document_error_code.value if document_error_code else None
        review_message = document_error_message
    elif student_unreadable:
        final_status = SubmissionStatus.NEEDS_REVIEW.value
        review_error_code = ErrorCode.STUDENT_NUMBER_UNREADABLE.value
        review_message = "Student number could not be read from the header"
    elif manual_review_count > 0:
        final_status = SubmissionStatus.NEEDS_REVIEW.value
        review_error_code = ErrorCode.ANSWER_LOW_CONFIDENCE.value
        review_message = f"{manual_review_count} answer(s) need manual review"

    with get_session() as session:
        SubmissionRepository(session).set_processing_completed(
            submission_id,
            score=result.score,
            total_points=result.total_points,
            percentage=result.percentage,
            status=final_status,
            error_code=review_error_code,
            error_message=review_message,
        )

    _log(
        submission_id,
        stage="pipeline",
        status="completed",
        message=f"score={result.score}/{result.total_points} ocr_accepted={ocr_accepted} escalated={escalated} manual_review={manual_review_count}",
    )

    return SubmissionOutcome(
        submission_id=submission_id,
        file_name=file_name,
        status=final_status,
        ocr_accepted_count=ocr_accepted,
        escalated_count=escalated,
        manual_review_count=manual_review_count,
    )


def process_batch(exam_id: int, submission_ids: list[int]) -> Iterator[tuple[BatchTally, SubmissionOutcome | None]]:
    """Process each submission in turn, yielding updated tallies for live UI
    progress. A single failed paper never stops the batch."""
    tally = BatchTally()

    for submission_id in submission_ids:
        try:
            outcome = process_submission(submission_id, exam_id)
        except Exception:
            logger.exception("Unhandled error processing submission %s", submission_id)
            with get_session() as session:
                SubmissionRepository(session).update_status(
                    submission_id,
                    SubmissionStatus.FAILED.value,
                    ErrorCode.DATABASE_ERROR.value,
                    "Unhandled processing error; see logs",
                )
            outcome = SubmissionOutcome(
                submission_id=submission_id,
                file_name=str(submission_id),
                status="failed",
                error_code=ErrorCode.DATABASE_ERROR.value,
            )

        tally.processed += 1
        tally.ocr_accepted_count += outcome.ocr_accepted_count
        tally.escalated_count += outcome.escalated_count
        if outcome.status == SubmissionStatus.COMPLETED.value:
            tally.successful += 1
        elif outcome.status == SubmissionStatus.NEEDS_REVIEW.value:
            tally.needs_review += 1
        else:
            tally.failed += 1
        tally.outcomes.append(outcome)

        yield tally, outcome
