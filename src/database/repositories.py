"""Repository helpers -- thin CRUD/query wrappers around SQLAlchemy models.

Each repository takes an explicit Session so services control transaction
boundaries (via database.get_session()).
"""

from __future__ import annotations

import datetime as dt

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.database.models import (
    Answer,
    Exam,
    ExamQuestion,
    ProcessingLog,
    Student,
    Submission,
)


class ExamRepository:
    def __init__(self, session: Session):
        self.session = session

    def create(
        self,
        name: str,
        course: str | None = None,
        description: str | None = None,
        date: dt.date | None = None,
        points_per_question: float = 1.0,
    ) -> Exam:
        exam = Exam(
            name=name,
            course=course,
            description=description,
            date=date,
            total_questions=0,
            points_per_question=points_per_question,
        )
        self.session.add(exam)
        self.session.flush()
        return exam

    def get(self, exam_id: int) -> Exam | None:
        return self.session.get(Exam, exam_id)

    def update(
        self,
        exam_id: int,
        name: str,
        course: str | None = None,
        description: str | None = None,
        date: dt.date | None = None,
        points_per_question: float = 1.0,
    ) -> Exam:
        exam = self.get(exam_id)
        if exam is None:
            raise ValueError(f"Exam {exam_id} not found")
        exam.name = name
        exam.course = course
        exam.description = description
        exam.date = date
        exam.points_per_question = points_per_question
        self.session.flush()
        return exam

    def delete(self, exam_id: int) -> None:
        exam = self.get(exam_id)
        if exam is None:
            raise ValueError(f"Exam {exam_id} not found")
        self.session.delete(exam)
        self.session.flush()

    def list_all(self) -> list[Exam]:
        return list(self.session.execute(select(Exam).order_by(Exam.created_at.desc())).scalars())

    def set_answer_key(self, exam_id: int, rows: list[tuple[int, str, float]]) -> None:
        """Replace the answer key for an exam with (question_number, answer, points) rows."""
        exam = self.get(exam_id)
        if exam is None:
            raise ValueError(f"Exam {exam_id} not found")

        existing = {
            q.question_number: q
            for q in self.session.execute(
                select(ExamQuestion).where(ExamQuestion.exam_id == exam_id)
            ).scalars()
        }
        seen_numbers: set[int] = set()
        for question_number, answer, points in rows:
            seen_numbers.add(question_number)
            existing_q = existing.get(question_number)
            if existing_q is not None:
                existing_q.correct_answer = answer
                existing_q.points = points
            else:
                self.session.add(
                    ExamQuestion(
                        exam_id=exam_id,
                        question_number=question_number,
                        correct_answer=answer,
                        points=points,
                    )
                )

        for question_number, q in existing.items():
            if question_number not in seen_numbers:
                self.session.delete(q)

        exam.total_questions = len(rows)
        self.session.flush()

    def get_answer_key(self, exam_id: int) -> list[ExamQuestion]:
        return list(
            self.session.execute(
                select(ExamQuestion)
                .where(ExamQuestion.exam_id == exam_id)
                .order_by(ExamQuestion.question_number)
            ).scalars()
        )


class StudentRepository:
    def __init__(self, session: Session):
        self.session = session

    def get_or_create(self, student_number: str, name: str | None = None) -> Student:
        student = self.session.execute(
            select(Student).where(Student.student_number == student_number)
        ).scalar_one_or_none()
        if student is not None:
            if name and not student.name:
                student.name = name
                self.session.flush()
            return student
        student = Student(student_number=student_number, name=name)
        self.session.add(student)
        self.session.flush()
        return student

    def get(self, student_id: int) -> Student | None:
        return self.session.get(Student, student_id)

    def list_all(self) -> list[Student]:
        return list(self.session.execute(select(Student).order_by(Student.name)).scalars())

    def update(self, student_id: int, student_number: str, name: str | None) -> Student:
        student = self.get(student_id)
        if student is None:
            raise ValueError(f"Student {student_id} not found")
        student.student_number = student_number
        student.name = name
        self.session.flush()
        return student

    def delete(self, student_id: int) -> None:
        """Deletes the student and unlinks (does not delete) their submissions,
        since a submission's exam/image/answers remain valid evidence on their
        own -- only the student association is lost."""
        student = self.get(student_id)
        if student is None:
            raise ValueError(f"Student {student_id} not found")
        for submission in self.session.execute(
            select(Submission).where(Submission.student_id == student_id)
        ).scalars():
            submission.student_id = None
        self.session.delete(student)
        self.session.flush()


class SubmissionRepository:
    def __init__(self, session: Session):
        self.session = session

    def create(
        self,
        exam_id: int,
        original_image_path: str,
        file_hash: str,
        student_id: int | None = None,
    ) -> Submission:
        submission = Submission(
            exam_id=exam_id,
            student_id=student_id,
            original_image_path=original_image_path,
            file_hash=file_hash,
            status="pending",
        )
        self.session.add(submission)
        self.session.flush()
        return submission

    def get(self, submission_id: int) -> Submission | None:
        return self.session.get(Submission, submission_id)

    def find_by_hash(self, exam_id: int, file_hash: str) -> Submission | None:
        return self.session.execute(
            select(Submission).where(
                Submission.exam_id == exam_id, Submission.file_hash == file_hash
            )
        ).scalar_one_or_none()

    def list_for_exam(self, exam_id: int) -> list[Submission]:
        return list(
            self.session.execute(
                select(Submission)
                .where(Submission.exam_id == exam_id)
                .order_by(Submission.created_at)
            ).scalars()
        )

    def delete(self, submission_id: int) -> list[str]:
        """Deletes the submission row (answers/logs cascade) and returns the
        image file paths (original, processed, per-answer crops) so the
        caller can remove them from disk."""
        submission = self.get(submission_id)
        if submission is None:
            raise ValueError(f"Submission {submission_id} not found")

        paths = [submission.original_image_path]
        if submission.processed_image_path:
            paths.append(submission.processed_image_path)
        for answer in submission.answers:
            if answer.crop_image_path:
                paths.append(answer.crop_image_path)

        self.session.delete(submission)
        self.session.flush()
        return paths

    def update_status(
        self,
        submission_id: int,
        status: str,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> None:
        submission = self.get(submission_id)
        if submission is None:
            raise ValueError(f"Submission {submission_id} not found")
        submission.status = status
        submission.error_code = error_code
        submission.error_message = error_message
        self.session.flush()

    def set_processing_started(self, submission_id: int) -> None:
        submission = self.get(submission_id)
        if submission is None:
            raise ValueError(f"Submission {submission_id} not found")
        submission.processing_started_at = dt.datetime.utcnow()
        submission.status = "processing"
        self.session.flush()

    def set_processing_completed(
        self,
        submission_id: int,
        score: float,
        total_points: float,
        percentage: float,
        status: str = "completed",
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> None:
        """error_code/error_message are always written (None clears them) --
        callers must pass through whatever reason should remain visible."""
        submission = self.get(submission_id)
        if submission is None:
            raise ValueError(f"Submission {submission_id} not found")
        submission.processing_completed_at = dt.datetime.utcnow()
        submission.score = score
        submission.total_points = total_points
        submission.percentage = percentage
        submission.status = status
        submission.error_code = error_code
        submission.error_message = error_message
        self.session.flush()

    def set_processed_image_path(self, submission_id: int, path: str) -> None:
        submission = self.get(submission_id)
        if submission is None:
            raise ValueError(f"Submission {submission_id} not found")
        submission.processed_image_path = path
        self.session.flush()

    def set_student(self, submission_id: int, student_id: int) -> None:
        submission = self.get(submission_id)
        if submission is None:
            raise ValueError(f"Submission {submission_id} not found")
        submission.student_id = student_id
        self.session.flush()


class AnswerRepository:
    def __init__(self, session: Session):
        self.session = session

    def create(
        self,
        submission_id: int,
        question_number: int,
        detected_answer: str | None,
        correct_answer: str | None,
        confidence: float | None,
        detection_method: str,
        answer_state: str,
        review_status: str,
        is_correct: bool | None,
        question_id: int | None = None,
        crop_image_path: str | None = None,
    ) -> Answer:
        answer = Answer(
            submission_id=submission_id,
            question_id=question_id,
            question_number=question_number,
            detected_answer=detected_answer,
            correct_answer=correct_answer,
            confidence=confidence,
            is_correct=is_correct,
            detection_method=detection_method,
            review_status=review_status,
            crop_image_path=crop_image_path,
            answer_state=answer_state,
            original_detected_answer=detected_answer,
            original_detection_method=detection_method,
            original_confidence=confidence,
        )
        self.session.add(answer)
        self.session.flush()
        return answer

    def get(self, answer_id: int) -> Answer | None:
        return self.session.get(Answer, answer_id)

    def delete_for_submission(self, submission_id: int) -> None:
        """Clear prior answers before reprocessing a submission, so a
        force-reprocess doesn't accumulate duplicate rows per question."""
        for answer in self.list_for_submission(submission_id):
            self.session.delete(answer)
        self.session.flush()

    def list_for_submission(self, submission_id: int) -> list[Answer]:
        return list(
            self.session.execute(
                select(Answer)
                .where(Answer.submission_id == submission_id)
                .order_by(Answer.question_number)
            ).scalars()
        )

    def list_pending_review(self, exam_id: int | None = None) -> list[Answer]:
        stmt = select(Answer).join(Submission).where(Answer.review_status == "pending")
        if exam_id is not None:
            stmt = stmt.where(Submission.exam_id == exam_id)
        return list(self.session.execute(stmt.order_by(Answer.created_at)).scalars())

    def correct(
        self,
        answer_id: int,
        new_detected_answer: str | None,
        new_answer_state: str,
    ) -> Answer:
        """Apply a manual correction. Preserves the original detection for audit."""
        answer = self.get(answer_id)
        if answer is None:
            raise ValueError(f"Answer {answer_id} not found")

        answer.detected_answer = new_detected_answer
        answer.answer_state = new_answer_state
        answer.detection_method = "manual"
        answer.review_status = "corrected"

        if new_answer_state == "clear" and new_detected_answer is not None and answer.correct_answer:
            answer.is_correct = new_detected_answer == answer.correct_answer
        else:
            answer.is_correct = None

        self.session.flush()
        return answer

    def confirm(self, answer_id: int) -> Answer:
        answer = self.get(answer_id)
        if answer is None:
            raise ValueError(f"Answer {answer_id} not found")
        answer.review_status = "confirmed"
        self.session.flush()
        return answer


class ProcessingLogRepository:
    def __init__(self, session: Session):
        self.session = session

    def log(
        self,
        submission_id: int,
        stage: str,
        status: str,
        message: str | None = None,
        duration_ms: int | None = None,
    ) -> ProcessingLog:
        entry = ProcessingLog(
            submission_id=submission_id,
            stage=stage,
            status=status,
            message=message,
            duration_ms=duration_ms,
        )
        self.session.add(entry)
        self.session.flush()
        return entry

    def list_for_submission(self, submission_id: int) -> list[ProcessingLog]:
        return list(
            self.session.execute(
                select(ProcessingLog)
                .where(ProcessingLog.submission_id == submission_id)
                .order_by(ProcessingLog.created_at)
            ).scalars()
        )
