from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import select

from src.database.database import get_session
from src.database.models import Student, Submission
from src.database.repositories import AnswerRepository, ExamRepository, StudentRepository, SubmissionRepository
from src.schemas import VALID_ANSWERS
from src.services import processing_service

DOCUMENT_ERROR_CODES = {"DOCUMENT_NOT_FOUND", "DOCUMENT_TRANSFORM_FAILED"}

SUBMISSION_LEVEL_ERROR_CODES = {
    "DOCUMENT_NOT_FOUND",
    "DOCUMENT_TRANSFORM_FAILED",
    "STUDENT_NUMBER_UNREADABLE",
}


@dataclass
class ReviewItem:
    answer_id: int
    submission_id: int
    student_number: str | None
    student_name: str | None
    question_number: int
    detected_answer: str | None
    confidence: float | None
    detection_method: str
    answer_state: str
    crop_image_path: str | None


@dataclass
class SubmissionIssue:
    submission_id: int
    exam_id: int
    error_code: str | None
    error_message: str | None
    student_number: str | None
    student_name: str | None
    original_image_path: str
    processed_image_path: str | None


def list_submission_issues(exam_id: int | None = None) -> list[SubmissionIssue]:
    """Submissions flagged needs_review for a reason that isn't a single
    answer -- document detection failure or an unreadable student number."""
    with get_session() as session:
        stmt = select(Submission).where(
            Submission.status == "needs_review",
            Submission.error_code.in_(SUBMISSION_LEVEL_ERROR_CODES),
        )
        if exam_id is not None:
            stmt = stmt.where(Submission.exam_id == exam_id)

        submissions = list(session.execute(stmt).scalars())
        issues: list[SubmissionIssue] = []
        for submission in submissions:
            student = submission.student
            issues.append(
                SubmissionIssue(
                    submission_id=submission.id,
                    exam_id=submission.exam_id,
                    error_code=submission.error_code,
                    error_message=submission.error_message,
                    student_number=student.student_number if student else None,
                    student_name=student.name if student else None,
                    original_image_path=submission.original_image_path,
                    processed_image_path=submission.processed_image_path,
                )
            )
        return issues


def resolve_submission_issue(submission_id: int, student_number: str, name: str | None = None) -> None:
    """Manually assign the student number the OCR/vision pipeline couldn't
    read, then re-evaluate whether the submission can leave needs_review."""
    with get_session() as session:
        student_repo = StudentRepository(session)
        student = student_repo.get_or_create(student_number, name)

        submission_repo = SubmissionRepository(session)
        submission_repo.set_student(submission_id, student.id)

        submission = submission_repo.get(submission_id)
        if submission is not None and submission.error_code == "STUDENT_NUMBER_UNREADABLE":
            _finalize_submission_status(session, submission_id, cleared_error_code="STUDENT_NUMBER_UNREADABLE")


def dismiss_document_issue(submission_id: int) -> None:
    """Accept a submission's document-detection issue as-is (the reviewer
    has visually confirmed the already-detected answers look fine despite
    the perspective-correction failure), clearing it without reprocessing."""
    with get_session() as session:
        submission = SubmissionRepository(session).get(submission_id)
        if submission is not None and submission.error_code in DOCUMENT_ERROR_CODES:
            _finalize_submission_status(session, submission_id, cleared_error_code=submission.error_code)


def reprocess_submission_issue(submission_id: int, exam_id: int) -> processing_service.SubmissionOutcome:
    """Re-run the full pipeline for one submission from the Review page,
    e.g. after a document-detection failure, without needing a re-upload."""
    return processing_service.process_submission(submission_id, exam_id)


def update_submission_student(submission_id: int, student_number: str, name: str | None) -> None:
    """Reassign or rename a submission's student, edited directly from the
    Results summary table -- independent of any pending submission-level
    issue (unlike resolve_submission_issue, this always applies the given
    name rather than only filling in a blank one)."""
    with get_session() as session:
        submission_repo = SubmissionRepository(session)

        student = session.execute(
            select(Student).where(Student.student_number == student_number)
        ).scalar_one_or_none()
        if student is not None:
            student.name = name
            session.flush()
        else:
            student = StudentRepository(session).get_or_create(student_number, name)

        submission_repo.set_student(submission_id, student.id)

        submission = submission_repo.get(submission_id)
        if submission is not None and submission.error_code == "STUDENT_NUMBER_UNREADABLE":
            _finalize_submission_status(session, submission_id, cleared_error_code="STUDENT_NUMBER_UNREADABLE")


def list_review_queue(exam_id: int | None = None) -> list[ReviewItem]:
    with get_session() as session:
        answer_repo = AnswerRepository(session)
        answers = answer_repo.list_pending_review(exam_id)

        items: list[ReviewItem] = []
        for answer in answers:
            submission = answer.submission
            student = submission.student if submission else None
            items.append(
                ReviewItem(
                    answer_id=answer.id,
                    submission_id=answer.submission_id,
                    student_number=student.student_number if student else None,
                    student_name=student.name if student else None,
                    question_number=answer.question_number,
                    detected_answer=answer.detected_answer,
                    confidence=answer.confidence,
                    detection_method=answer.detection_method,
                    answer_state=answer.answer_state,
                    crop_image_path=answer.crop_image_path,
                )
            )
        return items


def correct_answer(answer_id: int, new_value: str) -> None:
    """new_value is 'A'|'B'|'C'|'D'|'UNKNOWN'."""
    if new_value == "UNKNOWN":
        detected_answer = None
        answer_state = "ambiguous"
    elif new_value in VALID_ANSWERS:
        detected_answer = new_value
        answer_state = "clear"
    else:
        raise ValueError(f"Invalid correction value: {new_value!r}")

    with get_session() as session:
        answer_repo = AnswerRepository(session)
        answer_repo.correct(answer_id, detected_answer, answer_state)
        _recompute_submission_totals(session, answer_id)


def confirm_answer(answer_id: int) -> None:
    with get_session() as session:
        answer_repo = AnswerRepository(session)
        answer_repo.confirm(answer_id)


def set_answer(submission_id: int, question_number: int, detected_answer: str | None) -> None:
    """Manually set a submission's answer for one question from the Results
    question-breakdown editor -- creates the answer row if that question
    wasn't previously detected, otherwise applies a correction."""
    detected_answer = detected_answer or None
    if detected_answer is not None and detected_answer not in VALID_ANSWERS:
        raise ValueError(f"Invalid answer: {detected_answer!r}")

    with get_session() as session:
        answer_repo = AnswerRepository(session)
        submission_repo = SubmissionRepository(session)
        exam_repo = ExamRepository(session)

        submission = submission_repo.get(submission_id)
        if submission is None:
            raise ValueError(f"Submission {submission_id} not found")

        answer_key = {q.question_number: q.correct_answer for q in exam_repo.get_answer_key(submission.exam_id)}
        correct_answer = answer_key.get(question_number)
        answer_state = "clear" if detected_answer is not None else "blank"

        existing = next(
            (a for a in answer_repo.list_for_submission(submission_id) if a.question_number == question_number),
            None,
        )
        if existing is not None:
            answer_repo.correct(existing.id, detected_answer, answer_state)
        else:
            is_correct = (detected_answer == correct_answer) if (detected_answer and correct_answer) else None
            answer_repo.create(
                submission_id=submission_id,
                question_number=question_number,
                detected_answer=detected_answer,
                correct_answer=correct_answer,
                confidence=1.0,
                detection_method="manual",
                answer_state=answer_state,
                review_status="confirmed",
                is_correct=is_correct,
            )

        _finalize_submission_status(session, submission_id)


def delete_answer(submission_id: int, question_number: int) -> None:
    """Remove a submission's answer row for one question (a row deleted from
    the Results question-breakdown editor)."""
    with get_session() as session:
        answer_repo = AnswerRepository(session)
        existing = next(
            (a for a in answer_repo.list_for_submission(submission_id) if a.question_number == question_number),
            None,
        )
        if existing is not None:
            answer_repo.delete(existing.id)
        _finalize_submission_status(session, submission_id)


def _recompute_submission_totals(session, answer_id: int) -> None:
    answer_repo = AnswerRepository(session)
    answer = answer_repo.get(answer_id)
    if answer is None:
        return
    _finalize_submission_status(session, answer.submission_id)


def _finalize_submission_status(session, submission_id: int, cleared_error_code: str | None = None) -> None:
    """Recompute score/status for a submission after a correction or a
    submission-level issue resolution. A submission-level error (document
    detection failure, unreadable student number) keeps the submission in
    needs_review even once every per-answer flag is cleared, unless it's the
    specific error just resolved by the caller."""
    answer_repo = AnswerRepository(session)
    submission_repo = SubmissionRepository(session)
    exam_repo = ExamRepository(session)

    submission = submission_repo.get(submission_id)
    if submission is None:
        return
    all_answers = answer_repo.list_for_submission(submission_id)

    # total_points must reflect the full exam answer key, not just the
    # questions that happen to have a detected Answer row -- otherwise an
    # undetected answer line would silently shrink the denominator.
    answer_key = exam_repo.get_answer_key(submission.exam_id)
    total_points = sum(q.points for q in answer_key)
    points_by_question = {q.question_number: q.points for q in answer_key}
    score = sum(
        points_by_question.get(a.question_number, 1.0) for a in all_answers if a.is_correct is True
    )
    percentage = (score / total_points * 100.0) if total_points > 0 else 0.0

    still_pending = any(a.review_status == "pending" for a in all_answers)

    remaining_error_code = submission.error_code
    remaining_error_message = submission.error_message
    if cleared_error_code is not None and remaining_error_code == cleared_error_code:
        remaining_error_code = None
        remaining_error_message = None

    unresolved_submission_issue = remaining_error_code in SUBMISSION_LEVEL_ERROR_CODES
    status = "needs_review" if (still_pending or unresolved_submission_issue) else "completed"
    if status == "completed":
        remaining_error_code = None
        remaining_error_message = None

    submission_repo.set_processing_completed(
        submission_id,
        score=score,
        total_points=total_points,
        percentage=percentage,
        status=status,
        error_code=remaining_error_code,
        error_message=remaining_error_message,
    )
