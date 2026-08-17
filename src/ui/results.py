from __future__ import annotations

import pandas as pd
import streamlit as st

from src.database.database import get_session
from src.database.repositories import SubmissionRepository
from src.services import exam_service, review_service
from src.ui._answer_editor import render_answers_editor
from src.ui._format import humanize


def render() -> None:
    st.title("Results")

    exams = exam_service.list_exams()
    if not exams:
        st.info("No exams yet.")
        return

    exam_options = {f"{e.name}": e.id for e in exams}
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
                    "status": humanize(s.status),
                    "submission_id": s.id,
                }
            )

    if not summary_rows:
        st.info("No answer sheets for this exam yet.")
        return

    summary_df = pd.DataFrame(summary_rows)
    summary_editor_key = f"summary_editor_{exam_id}"
    st.data_editor(
        summary_df,
        key=summary_editor_key,
        hide_index=True,
        num_rows="fixed",
        column_config={
            "student_number": st.column_config.TextColumn("Student number"),
            "name": st.column_config.TextColumn("Name"),
            "score": st.column_config.NumberColumn("Score", disabled=True),
            "total_points": st.column_config.NumberColumn("Total points", disabled=True),
            "percentage": st.column_config.NumberColumn("Percentage", format="%.1f%%", disabled=True),
            "status": st.column_config.TextColumn("Status", disabled=True),
            "submission_id": st.column_config.NumberColumn("Answer sheet ID", disabled=True),
        },
    )

    if st.button("Save answer sheets"):
        summary_editor_state = st.session_state[summary_editor_key]

        for idx, changes in summary_editor_state["edited_rows"].items():
            if "student_number" in changes or "name" in changes:
                original_row = summary_df.iloc[idx]
                student_number = changes.get("student_number", original_row["student_number"])
                name = changes.get("name", original_row["name"])
                if student_number:
                    review_service.update_submission_student(
                        int(original_row["submission_id"]), student_number, name or None
                    )

        st.toast("Answer sheets updated.", icon=":material/check_circle:")
        st.rerun()

    st.divider()
    st.subheader("Question breakdown")
    submission_labels = {f"{r['student_number']} (answer sheet {r['submission_id']})": r["submission_id"] for r in summary_rows}
    chosen = st.selectbox("Select answer sheet", list(submission_labels.keys()))
    submission_id = submission_labels[chosen]

    render_answers_editor(submission_id)
