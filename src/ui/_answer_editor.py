from __future__ import annotations

import pandas as pd
import streamlit as st

from src.database.database import get_session
from src.database.repositories import AnswerRepository
from src.schemas import VALID_ANSWERS
from src.services import review_service
from src.ui._format import humanize


def render_answers_editor(submission_id: int, *, key_prefix: str = "answers") -> None:
    """Editable question/detected-answer table for one submission: lets a
    user correct a detected letter, add a row for an undetected question, or
    delete one -- shared by the Results and Students pages."""
    with get_session() as session:
        answers = AnswerRepository(session).list_for_submission(submission_id)
        rows = [
            {
                "question": a.question_number,
                "detected": a.detected_answer,
                "correct_answer": a.correct_answer,
                "is_correct": a.is_correct,
                "state": humanize(a.answer_state),
                "review_status": humanize(a.review_status),
            }
            for a in answers
        ]

    answers_df = (
        pd.DataFrame(rows)
        if rows
        else pd.DataFrame(
            {
                "question": pd.Series(dtype="int"),
                "detected": pd.Series(dtype="string"),
                "correct_answer": pd.Series(dtype="string"),
                "is_correct": pd.Series(dtype="boolean"),
                "state": pd.Series(dtype="string"),
                "review_status": pd.Series(dtype="string"),
            }
        )
    )

    editor_key = f"{key_prefix}_editor_{submission_id}"
    st.data_editor(
        answers_df,
        key=editor_key,
        hide_index=True,
        num_rows="dynamic",
        column_config={
            "question": st.column_config.NumberColumn("Question", width="small"),
            "detected": st.column_config.SelectboxColumn("Detected answer", options=list(VALID_ANSWERS)),
            "correct_answer": st.column_config.TextColumn("Correct answer", disabled=True),
            "is_correct": st.column_config.CheckboxColumn("Correct?", disabled=True),
            "state": st.column_config.TextColumn("Answer state", disabled=True),
            "review_status": st.column_config.TextColumn("Review status", disabled=True),
        },
    )

    if st.button("Save answers", key=f"save_{key_prefix}_{submission_id}"):
        editor_state = st.session_state[editor_key]

        for idx, changes in editor_state["edited_rows"].items():
            if "detected" in changes:
                question_number = int(answers_df.iloc[idx]["question"])
                review_service.set_answer(submission_id, question_number, changes["detected"])

        for new_row in editor_state["added_rows"]:
            question_number = new_row.get("question")
            if question_number is not None:
                review_service.set_answer(submission_id, int(question_number), new_row.get("detected"))

        for idx in editor_state["deleted_rows"]:
            question_number = int(answers_df.iloc[idx]["question"])
            review_service.delete_answer(submission_id, question_number)

        st.toast("Answers updated.", icon=":material/check_circle:")
        st.rerun()
