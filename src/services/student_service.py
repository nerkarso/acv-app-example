from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

from src.database.database import get_session
from src.database.models import Student
from src.database.repositories import StudentRepository, SubmissionRepository


@dataclass
class StudentSubmission:
    submission_id: int
    exam_id: int
    exam_name: str
    status: str
    score: float | None
    total_points: float | None
    percentage: float | None
    original_image_path: str
    processed_image_path: str | None
    created_at: dt.datetime


def list_submissions_for_student(student_id: int) -> list[StudentSubmission]:
    with get_session() as session:
        submissions = SubmissionRepository(session).list_for_student(student_id)
        return [
            StudentSubmission(
                submission_id=s.id,
                exam_id=s.exam_id,
                exam_name=s.exam.name,
                status=s.status,
                score=s.score,
                total_points=s.total_points,
                percentage=s.percentage,
                original_image_path=s.original_image_path,
                processed_image_path=s.processed_image_path,
                created_at=s.created_at,
            )
            for s in submissions
        ]


def list_students() -> list[Student]:
    with get_session() as session:
        students = StudentRepository(session).list_all()
        session.expunge_all()
        return students


def get_student(student_id: int) -> Student | None:
    with get_session() as session:
        student = StudentRepository(session).get(student_id)
        if student is not None:
            session.expunge(student)
        return student


def create_student(student_number: str, name: str | None = None) -> int:
    with get_session() as session:
        student = StudentRepository(session).get_or_create(student_number, name)
        return student.id


def update_student(student_id: int, student_number: str, name: str | None) -> None:
    with get_session() as session:
        StudentRepository(session).update(student_id, student_number, name)


def delete_student(student_id: int) -> None:
    with get_session() as session:
        StudentRepository(session).delete(student_id)
