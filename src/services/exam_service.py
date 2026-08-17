from __future__ import annotations

import csv
import datetime as dt
import io
import re
from pathlib import Path

from src.database.database import get_session
from src.database.models import Exam, ExamQuestion, Submission
from src.database.repositories import ExamRepository, SubmissionRepository
from src.schemas import AnswerKeyRow


def create_exam(
    name: str,
    course: str | None = None,
    description: str | None = None,
    date: dt.date | None = None,
    points_per_question: float = 1.0,
) -> int:
    with get_session() as session:
        repo = ExamRepository(session)
        exam = repo.create(
            name=name,
            course=course,
            description=description,
            date=date,
            points_per_question=points_per_question,
        )
        return exam.id


def list_exams() -> list[Exam]:
    with get_session() as session:
        repo = ExamRepository(session)
        exams = repo.list_all()
        session.expunge_all()
        return exams


def get_exam(exam_id: int) -> Exam | None:
    with get_session() as session:
        repo = ExamRepository(session)
        exam = repo.get(exam_id)
        if exam is not None:
            session.expunge(exam)
        return exam


def update_exam(
    exam_id: int,
    name: str,
    course: str | None = None,
    description: str | None = None,
    date: dt.date | None = None,
    points_per_question: float = 1.0,
) -> None:
    with get_session() as session:
        repo = ExamRepository(session)
        repo.update(
            exam_id,
            name=name,
            course=course,
            description=description,
            date=date,
            points_per_question=points_per_question,
        )


def delete_exam(exam_id: int) -> None:
    """Deletes the exam and everything under it (questions, submissions,
    answers, processing logs cascade via the ORM relationships), and best-
    effort removes the submissions' image files from disk."""
    with get_session() as session:
        submission_repo = SubmissionRepository(session)
        exam_repo = ExamRepository(session)

        image_paths: list[str] = []
        for submission in submission_repo.list_for_exam(exam_id):
            if submission.original_image_path:
                image_paths.append(submission.original_image_path)
            if submission.processed_image_path:
                image_paths.append(submission.processed_image_path)
            for answer in submission.answers:
                if answer.crop_image_path:
                    image_paths.append(answer.crop_image_path)

        exam_repo.delete(exam_id)

    for path_str in image_paths:
        try:
            Path(path_str).unlink(missing_ok=True)
        except OSError:
            pass


def delete_submission(submission_id: int) -> None:
    with get_session() as session:
        paths = SubmissionRepository(session).delete(submission_id)

    for path_str in paths:
        try:
            Path(path_str).unlink(missing_ok=True)
        except OSError:
            pass


def set_answer_key_from_dropdowns(exam_id: int, answers: dict[int, str], points: float = 1.0) -> None:
    """answers: {question_number: 'A'|'B'|'C'|'D'}"""
    rows = [(qnum, letter, points) for qnum, letter in sorted(answers.items())]
    with get_session() as session:
        repo = ExamRepository(session)
        repo.set_answer_key(exam_id, rows)


def set_answer_key_from_csv(exam_id: int, csv_bytes: bytes) -> list[str]:
    """CSV columns: question,answer,points. Returns a list of row-level warnings."""
    warnings: list[str] = []
    reader = csv.DictReader(io.StringIO(csv_bytes.decode("utf-8-sig")))
    parsed_rows: list[tuple[int, str, float]] = []

    for i, row in enumerate(reader, start=2):  # header is row 1
        try:
            question = int(row["question"])
            points = float(row.get("points") or 1.0)
            key_row = AnswerKeyRow(question=question, answer=row["answer"], points=points)
            parsed_rows.append((key_row.question, key_row.answer, key_row.points))
        except Exception as exc:
            warnings.append(f"Row {i}: {exc}")

    with get_session() as session:
        repo = ExamRepository(session)
        repo.set_answer_key(exam_id, parsed_rows)

    return warnings


def set_answer_key_from_text(exam_id: int, text: str, points: float = 1.0) -> list[str]:
    """Parse pasted answer-key text: either full CSV (question,answer,points
    header) or a plain list of answer letters, one per question in order."""
    text = text.strip()
    first_line = text.splitlines()[0] if text else ""
    if "," in first_line and re.search(r"(?i)\bquestion\b", first_line):
        return set_answer_key_from_csv(exam_id, text.encode("utf-8"))

    warnings: list[str] = []
    parsed_rows: list[tuple[int, str, float]] = []
    for i, letter in enumerate(re.findall(r"[A-Da-d]", text), start=1):
        try:
            key_row = AnswerKeyRow(question=i, answer=letter, points=points)
            parsed_rows.append((key_row.question, key_row.answer, key_row.points))
        except Exception as exc:
            warnings.append(f"Answer {i}: {exc}")

    with get_session() as session:
        repo = ExamRepository(session)
        repo.set_answer_key(exam_id, parsed_rows)

    return warnings


def get_answer_key(exam_id: int) -> list[ExamQuestion]:
    with get_session() as session:
        repo = ExamRepository(session)
        questions = repo.get_answer_key(exam_id)
        session.expunge_all()
        return questions


def list_submissions(exam_id: int) -> list[Submission]:
    with get_session() as session:
        repo = SubmissionRepository(session)
        submissions = repo.list_for_exam(exam_id)
        session.expunge_all()
        return submissions
