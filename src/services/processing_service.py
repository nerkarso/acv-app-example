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
from src.ocr.paddle import run_ocr, run_ocr_single_line
from src.ocr.validation import STUDENT_NUMBER_PATTERN, parse_answer_line, validate_header_field
from src.schemas import (
    AnswerState,
    DetectionMethod,
    ErrorCode,
    ReviewStatus,
    SubmissionStatus,
)
from src.vision.document_detection import detect_document_corners
from src.vision.line_detection import crop_region, detect_answer_line_regions, detect_header_field_regions
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


def _extract_header_fields(image, submission_id: int) -> dict[str, str | None]:
    full_page_lines = run_ocr(image)
    text_boxes = [(line.text, line.bbox) for line in full_page_lines]

    regions = detect_header_field_regions(image, text_boxes, HEADER_LABEL_KEYWORDS)
    fields: dict[str, str | None] = {}

    for region in regions:
        crop = crop_region(image, region.bbox)
        raw = run_ocr_single_line(crop)
        pattern = STUDENT_NUMBER_PATTERN if region.field_name == "student_number" else None
        result = validate_header_field(raw, region.field_name, pattern)
        fields[region.field_name] = result.parsed_value
        if result.needs_escalation:
            _log(
                submission_id,
                stage="header_extraction",
                status="needs_review",
                message=f"{region.field_name}: low confidence or unparseable ('{result.raw_text}')",
            )

    return fields


def _process_answers(
    image,
    exam_id: int,
    submission_id: int,
    num_questions: int,
) -> tuple[list[DetectedAnswer], int, int, int]:
    """Returns (detected_answers, ocr_accepted_count, escalated_count, manual_review_count)."""
    provider = get_vision_provider()
    regions = detect_answer_line_regions(image)

    detected: list[DetectedAnswer] = []
    ocr_accepted = 0
    escalated = 0
    manual_review = 0

    answer_rows: list[dict] = []

    for region in regions:
        crop = crop_region(image, region.bbox)
        crop_path = settings.crops_path / f"{submission_id}_{uuid.uuid4().hex}.png"

        raw = run_ocr_single_line(crop)
        strike = detect_strike_through(crop)

        if raw is None:
            ocr_result = None
        else:
            ocr_result = parse_answer_line(raw, num_questions)

        if ocr_result is None or ocr_result.question_number is None:
            # Can't anchor this crop to a question number at all -- skip, it
            # is likely not an answer line (e.g. header text).
            continue

        question_number = ocr_result.question_number
        confidence = ocr_result.confidence
        detected_answer = ocr_result.detected_answer
        method = DetectionMethod.PADDLEOCR
        review_status = ReviewStatus.NOT_REQUIRED
        answer_state = AnswerState.CLEAR

        if strike.is_struck_through:
            answer_state = AnswerState.STRUCK_THROUGH
            review_status = ReviewStatus.PENDING
            manual_review += 1
        elif detected_answer is None:
            answer_state = AnswerState.BLANK if not raw.text.strip() else AnswerState.AMBIGUOUS

        if answer_state == AnswerState.CLEAR:
            if confidence >= settings.confidence_auto_accept and detected_answer is not None:
                ocr_accepted += 1
            elif confidence >= settings.confidence_escalate_min and provider.is_available:
                detection = provider.classify_answer(crop, ["A", "B", "C", "D"])
                escalated += 1
                detected_answer = detection.answer
                confidence = detection.confidence
                method = (
                    DetectionMethod.CLOUD_VLM
                    if settings.vision_provider == "claude"
                    else DetectionMethod.LOCAL_VLM
                )
                answer_state = detection.state
                if detection.answer is None:
                    review_status = ReviewStatus.PENDING
                    manual_review += 1
            else:
                review_status = ReviewStatus.PENDING
                manual_review += 1

        try:
            cv2.imwrite(str(crop_path), crop)
            crop_path_str = str(crop_path)
        except Exception:
            crop_path_str = None

        answer_rows.append(
            {
                "question_number": question_number,
                "detected_answer": detected_answer,
                "confidence": confidence,
                "detection_method": method.value,
                "answer_state": answer_state.value,
                "review_status": review_status.value,
                "crop_image_path": crop_path_str,
            }
        )

        detected.append(
            DetectedAnswer(
                question_number=question_number,
                detected_answer=detected_answer,
                answer_state=answer_state,
            )
        )

    with get_session() as session:
        exam_repo = ExamRepository(session)
        answer_repo = AnswerRepository(session)
        key_by_number = {q.question_number: q for q in exam_repo.get_answer_key(exam_id)}

        seen_numbers: set[int] = set()
        for row in answer_rows:
            qnum = row["question_number"]
            if qnum in seen_numbers:
                continue  # keep first detection per question number
            seen_numbers.add(qnum)
            key_entry = key_by_number.get(qnum)
            answer_repo.create(
                submission_id=submission_id,
                question_number=qnum,
                detected_answer=row["detected_answer"],
                correct_answer=key_entry.correct_answer if key_entry else None,
                confidence=row["confidence"],
                detection_method=row["detection_method"],
                answer_state=row["answer_state"],
                review_status=row["review_status"],
                is_correct=None,
                question_id=key_entry.id if key_entry else None,
                crop_image_path=row["crop_image_path"],
            )

    return detected, ocr_accepted, escalated, manual_review


def process_submission(submission_id: int, exam_id: int) -> SubmissionOutcome:
    with get_session() as session:
        SubmissionRepository(session).set_processing_started(submission_id)

    stale_files: list[str] = []
    with get_session() as session:
        submission = SubmissionRepository(session).get(submission_id)
        original_path = submission.original_image_path
        exam = ExamRepository(session).get(exam_id)
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
    if not validate_image(image):
        return fail(ErrorCode.INVALID_IMAGE, "Image could not be read or is corrupt")

    image = resize_if_oversized(image)

    detection = detect_document_corners(image)
    document_ok = detection.success
    working_image = image

    if not document_ok:
        needs_review(ErrorCode.DOCUMENT_NOT_FOUND, "Document contour not found; using original image")
    else:
        warped = warp_document(image, detection.corners)
        if warped is None:
            document_ok = False
            needs_review(ErrorCode.DOCUMENT_TRANSFORM_FAILED, "Perspective transform failed; using original image")
        else:
            working_image = warped

    working_image = normalize_resolution(working_image)
    working_image = apply_clahe(working_image)

    processed_path = settings.processed_path / f"{submission_id}_{uuid.uuid4().hex}.png"
    cv2.imwrite(str(processed_path), working_image)
    with get_session() as session:
        SubmissionRepository(session).set_processed_image_path(submission_id, str(processed_path))

    header_fields = _extract_header_fields(working_image, submission_id)
    student_number = header_fields.get("student_number")
    name = header_fields.get("name")

    student_unreadable = not student_number
    if student_unreadable:
        _log(submission_id, stage="header_extraction", status="needs_review", message="STUDENT_NUMBER_UNREADABLE")
    else:
        with get_session() as session:
            student_repo = StudentRepository(session)
            student = student_repo.get_or_create(student_number, name)
            SubmissionRepository(session).set_student(submission_id, student.id)

    detected_answers, ocr_accepted, escalated, manual_review_count = _process_answers(
        working_image, exam_id, submission_id, num_questions
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
        # error_code/message from the document-detection failure above are
        # already persisted on the submission; leave them as the reason.
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
