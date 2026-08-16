from __future__ import annotations

import datetime as dt

from sqlalchemy import (
    Float,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


def _utcnow() -> dt.datetime:
    return dt.datetime.utcnow()


class Exam(Base):
    __tablename__ = "exams"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    course: Mapped[str | None] = mapped_column(String, nullable=True)
    description: Mapped[str | None] = mapped_column(String, nullable=True)
    date: Mapped[dt.date | None] = mapped_column(nullable=True)
    total_questions: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    points_per_question: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    created_at: Mapped[dt.datetime] = mapped_column(default=_utcnow)
    updated_at: Mapped[dt.datetime] = mapped_column(default=_utcnow, onupdate=_utcnow)

    questions: Mapped[list["ExamQuestion"]] = relationship(
        back_populates="exam", cascade="all, delete-orphan"
    )
    submissions: Mapped[list["Submission"]] = relationship(
        back_populates="exam", cascade="all, delete-orphan"
    )


class ExamQuestion(Base):
    __tablename__ = "exam_questions"
    __table_args__ = (UniqueConstraint("exam_id", "question_number", name="uq_exam_question_number"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    exam_id: Mapped[int] = mapped_column(ForeignKey("exams.id"), nullable=False)
    question_number: Mapped[int] = mapped_column(Integer, nullable=False)
    correct_answer: Mapped[str] = mapped_column(String, nullable=False)  # A|B|C|D
    points: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    created_at: Mapped[dt.datetime] = mapped_column(default=_utcnow)

    exam: Mapped["Exam"] = relationship(back_populates="questions")
    answers: Mapped[list["Answer"]] = relationship(back_populates="question")


class Student(Base):
    __tablename__ = "students"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    student_number: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    name: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(default=_utcnow)
    updated_at: Mapped[dt.datetime] = mapped_column(default=_utcnow, onupdate=_utcnow)

    submissions: Mapped[list["Submission"]] = relationship(back_populates="student")


class Submission(Base):
    __tablename__ = "submissions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    exam_id: Mapped[int] = mapped_column(ForeignKey("exams.id"), nullable=False)
    student_id: Mapped[int | None] = mapped_column(ForeignKey("students.id"), nullable=True)

    original_image_path: Mapped[str] = mapped_column(String, nullable=False)
    processed_image_path: Mapped[str | None] = mapped_column(String, nullable=True)
    file_hash: Mapped[str] = mapped_column(String, nullable=False, index=True)

    status: Mapped[str] = mapped_column(String, nullable=False, default="pending")
    error_code: Mapped[str | None] = mapped_column(String, nullable=True)
    error_message: Mapped[str | None] = mapped_column(String, nullable=True)

    score: Mapped[float | None] = mapped_column(Float, nullable=True)
    total_points: Mapped[float | None] = mapped_column(Float, nullable=True)
    percentage: Mapped[float | None] = mapped_column(Float, nullable=True)

    processing_started_at: Mapped[dt.datetime | None] = mapped_column(nullable=True)
    processing_completed_at: Mapped[dt.datetime | None] = mapped_column(nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(default=_utcnow)
    updated_at: Mapped[dt.datetime] = mapped_column(default=_utcnow, onupdate=_utcnow)

    exam: Mapped["Exam"] = relationship(back_populates="submissions")
    student: Mapped["Student | None"] = relationship(back_populates="submissions")
    answers: Mapped[list["Answer"]] = relationship(
        back_populates="submission", cascade="all, delete-orphan"
    )
    logs: Mapped[list["ProcessingLog"]] = relationship(
        back_populates="submission", cascade="all, delete-orphan"
    )


class Answer(Base):
    __tablename__ = "answers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    submission_id: Mapped[int] = mapped_column(ForeignKey("submissions.id"), nullable=False)
    question_id: Mapped[int | None] = mapped_column(ForeignKey("exam_questions.id"), nullable=True)
    question_number: Mapped[int] = mapped_column(Integer, nullable=False)

    detected_answer: Mapped[str | None] = mapped_column(String, nullable=True)  # A|B|C|D|NULL
    correct_answer: Mapped[str | None] = mapped_column(String, nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    is_correct: Mapped[bool | None] = mapped_column(nullable=True)

    detection_method: Mapped[str] = mapped_column(String, nullable=False)
    review_status: Mapped[str] = mapped_column(String, nullable=False, default="not_required")
    crop_image_path: Mapped[str | None] = mapped_column(String, nullable=True)
    answer_state: Mapped[str] = mapped_column(String, nullable=False, default="clear")

    # Audit trail: original detection preserved even after manual correction
    original_detected_answer: Mapped[str | None] = mapped_column(String, nullable=True)
    original_detection_method: Mapped[str | None] = mapped_column(String, nullable=True)
    original_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)

    created_at: Mapped[dt.datetime] = mapped_column(default=_utcnow)
    updated_at: Mapped[dt.datetime] = mapped_column(default=_utcnow, onupdate=_utcnow)

    submission: Mapped["Submission"] = relationship(back_populates="answers")
    question: Mapped["ExamQuestion | None"] = relationship(back_populates="answers")


class ProcessingLog(Base):
    __tablename__ = "processing_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    submission_id: Mapped[int] = mapped_column(ForeignKey("submissions.id"), nullable=False)
    stage: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False)
    message: Mapped[str | None] = mapped_column(String, nullable=True)
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(default=_utcnow)

    submission: Mapped["Submission"] = relationship(back_populates="logs")
