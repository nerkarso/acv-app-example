from __future__ import annotations

import io
import statistics
from dataclasses import dataclass

import pandas as pd
from sqlalchemy import select

from src.database.database import get_session
from src.database.models import Answer, Exam, Submission


@dataclass
class DashboardStats:
    total_exams: int
    total_students: int
    average_score_pct: float | None
    pass_rate_pct: float | None
    exams_requiring_review: int


PASS_THRESHOLD_PCT = 60.0


def get_dashboard_stats() -> DashboardStats:
    with get_session() as session:
        exams = list(session.execute(select(Exam)).scalars())
        submissions = list(
            session.execute(select(Submission).where(Submission.percentage.is_not(None))).scalars()
        )
        student_ids = {s.student_id for s in submissions if s.student_id is not None}

        percentages = [s.percentage for s in submissions if s.percentage is not None]
        average = statistics.mean(percentages) if percentages else None
        pass_rate = (
            (sum(1 for p in percentages if p >= PASS_THRESHOLD_PCT) / len(percentages) * 100.0)
            if percentages
            else None
        )
        needing_review = len({s.exam_id for s in submissions if s.status == "needs_review"})

        return DashboardStats(
            total_exams=len(exams),
            total_students=len(student_ids),
            average_score_pct=average,
            pass_rate_pct=pass_rate,
            exams_requiring_review=needing_review,
        )


RESULTS_COLUMNS = [
    "submission_id",
    "student_number",
    "name",
    "score",
    "total_points",
    "percentage",
    "status",
]


def get_results_dataframe(exam_id: int) -> pd.DataFrame:
    with get_session() as session:
        submissions = list(
            session.execute(select(Submission).where(Submission.exam_id == exam_id)).scalars()
        )
        rows = []
        for submission in submissions:
            student = submission.student
            rows.append(
                {
                    "submission_id": submission.id,
                    "student_number": student.student_number if student else None,
                    "name": student.name if student else None,
                    "score": submission.score,
                    "total_points": submission.total_points,
                    "percentage": submission.percentage,
                    "status": submission.status,
                }
            )
        return pd.DataFrame(rows, columns=RESULTS_COLUMNS)


def get_grade_distribution(exam_id: int) -> dict:
    df = get_results_dataframe(exam_id)
    percentages = df["percentage"].dropna().tolist()
    if not percentages:
        return {"percentages": [], "mean": None, "median": None, "min": None, "max": None, "stdev": None, "pass_rate": None}

    return {
        "percentages": percentages,
        "mean": statistics.mean(percentages),
        "median": statistics.median(percentages),
        "min": min(percentages),
        "max": max(percentages),
        "stdev": statistics.stdev(percentages) if len(percentages) > 1 else 0.0,
        "pass_rate": sum(1 for p in percentages if p >= PASS_THRESHOLD_PCT) / len(percentages) * 100.0,
    }


def get_question_difficulty(exam_id: int) -> pd.DataFrame:
    with get_session() as session:
        answers = list(
            session.execute(
                select(Answer).join(Submission).where(Submission.exam_id == exam_id)
            ).scalars()
        )

        by_question: dict[int, list[bool | None]] = {}
        for answer in answers:
            by_question.setdefault(answer.question_number, []).append(answer.is_correct)

        rows = []
        for question_number, results in by_question.items():
            gradable = [r for r in results if r is not None]
            pct_correct = (sum(1 for r in gradable if r) / len(gradable) * 100.0) if gradable else None
            rows.append(
                {
                    "question_number": question_number,
                    "pct_correct": pct_correct,
                    "n_gradable": len(gradable),
                    "n_ungraded": len(results) - len(gradable),
                }
            )

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("pct_correct", ascending=True, na_position="last")
        return df


def get_course_trend(course: str) -> pd.DataFrame:
    with get_session() as session:
        exams = list(session.execute(select(Exam).where(Exam.course == course)).scalars())
        rows = []
        for exam in exams:
            submissions = list(
                session.execute(
                    select(Submission).where(
                        Submission.exam_id == exam.id, Submission.percentage.is_not(None)
                    )
                ).scalars()
            )
            percentages = [s.percentage for s in submissions if s.percentage is not None]
            rows.append(
                {
                    "exam_id": exam.id,
                    "exam_name": exam.name,
                    "date": exam.date,
                    "average_pct": statistics.mean(percentages) if percentages else None,
                    "n_submissions": len(submissions),
                }
            )
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("date")
        return df


def get_ocr_vs_escalation_breakdown(exam_id: int | None = None) -> pd.DataFrame:
    with get_session() as session:
        stmt = select(Answer)
        if exam_id is not None:
            stmt = stmt.join(Submission).where(Submission.exam_id == exam_id)
        answers = list(session.execute(stmt).scalars())

        counts: dict[str, int] = {}
        for answer in answers:
            counts[answer.detection_method] = counts.get(answer.detection_method, 0) + 1

        return pd.DataFrame(
            [{"detection_method": method, "count": count} for method, count in counts.items()]
        )


def export_results_csv(exam_id: int) -> bytes:
    df = _build_export_dataframe(exam_id)
    return df.to_csv(index=False).encode("utf-8")


def export_results_xlsx(exam_id: int) -> bytes:
    df = _build_export_dataframe(exam_id)
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Results")
    return buffer.getvalue()


def export_question_analysis_csv(exam_id: int) -> bytes:
    df = get_question_difficulty(exam_id)
    return df.to_csv(index=False).encode("utf-8")


def export_question_analysis_xlsx(exam_id: int) -> bytes:
    df = get_question_difficulty(exam_id)
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Question Analysis")
    return buffer.getvalue()


def _build_export_dataframe(exam_id: int) -> pd.DataFrame:
    with get_session() as session:
        submissions = list(
            session.execute(select(Submission).where(Submission.exam_id == exam_id)).scalars()
        )

        rows = []
        for submission in submissions:
            student = submission.student
            answers = sorted(submission.answers, key=lambda a: a.question_number)
            row = {
                "student_number": student.student_number if student else None,
                "name": student.name if student else None,
                "score": submission.score,
                "total_points": submission.total_points,
                "percentage": submission.percentage,
                "status": submission.status,
            }
            for answer in answers:
                row[f"q{answer.question_number}"] = answer.detected_answer
            rows.append(row)

        return pd.DataFrame(rows)
