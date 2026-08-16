from __future__ import annotations

import pandas as pd
import streamlit as st

from src.database.database import get_session
from src.database.repositories import AnswerRepository, SubmissionRepository
from src.services import exam_service


def render() -> None:
    st.title("Results")

    exams = exam_service.list_exams()
    if not exams:
        st.info("No exams yet.")
        return

    exam_options = {f"{e.name} (id={e.id})": e.id for e in exams}
    selected_label = st.selectbox("Exam", list(exam_options.keys()))
    exam_id = exam_options[selected_label]

    summary_rows = []
    with get_session() as session:
        submissions = SubmissionRepository(session).list_for_exam(exam_id)
        for s in submissions:
            student = s.student
            summary_rows.append(
                {
                    "student_number": student.student_number if student else "unknown",
                    "name": student.name if student else "",
                    "score": s.score,
                    "total_points": s.total_points,
                    "percentage": s.percentage,
                    "status": s.status,
                    "submission_id": s.id,
                }
            )

    if not summary_rows:
        st.info("No submissions for this exam yet.")
        return

    st.dataframe(pd.DataFrame(summary_rows), hide_index=True)

    st.divider()
    st.subheader("Question breakdown")
    submission_labels = {f"{r['student_number']} (submission {r['submission_id']})": r["submission_id"] for r in summary_rows}
    chosen = st.selectbox("Select submission", list(submission_labels.keys()))
    submission_id = submission_labels[chosen]

    with get_session() as session:
        answer_repo = AnswerRepository(session)
        answers = answer_repo.list_for_submission(submission_id)
        rows = [
            {
                "question": a.question_number,
                "detected": a.detected_answer,
                "correct_answer": a.correct_answer,
                "is_correct": a.is_correct,
                "state": a.answer_state,
                "review_status": a.review_status,
            }
            for a in answers
        ]

    st.dataframe(pd.DataFrame(rows), hide_index=True)
