from __future__ import annotations

from src.database.database import get_session
from src.database.models import Student
from src.database.repositories import StudentRepository


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
