from __future__ import annotations

import datetime as dt

import pandas as pd
import streamlit as st

from src.schemas import VALID_ANSWERS
from src.services import exam_service


def _create_exam_form() -> None:
    with st.expander("Create exam", icon=":material/add:"):
        with st.form("create_exam_form"):
            name = st.text_input("Exam name")
            course = st.text_input("Course")
            description = st.text_area("Description", height=80)
            date = st.date_input("Date", value=dt.date.today())
            total_questions = st.number_input("Number of questions", min_value=1, max_value=200, value=20)
            points_per_question = st.number_input("Points per question", min_value=0.1, value=1.0, step=0.5)

            submitted = st.form_submit_button("Create exam")
            if submitted:
                if not name.strip():
                    st.error("Exam name is required.")
                    return
                exam_id = exam_service.create_exam(
                    name=name.strip(),
                    course=course.strip() or None,
                    description=description.strip() or None,
                    date=date,
                    points_per_question=points_per_question,
                )
                st.session_state["_pending_answer_key_size"] = int(total_questions)
                st.session_state["selected_exam_id"] = exam_id
                st.success(f"Exam '{name}' created (id={exam_id}). Define the answer key below.")
                st.rerun()


@st.dialog("Delete exam")
def _confirm_delete_exam(exam_id: int, exam_name: str) -> None:
    st.warning(
        f"This permanently deletes '{exam_name}' and all of its questions, submissions, "
        "answers, and images. This cannot be undone."
    )
    col1, col2 = st.columns(2)
    if col1.button("Delete", type="primary"):
        exam_service.delete_exam(exam_id)
        st.session_state.pop("selected_exam_id", None)
        st.rerun()
    if col2.button("Cancel"):
        st.rerun()


def _exam_details_editor(exam) -> None:
    st.subheader("Exam details")
    with st.form("edit_exam_form"):
        name = st.text_input("Exam name", value=exam.name)
        course = st.text_input("Course", value=exam.course or "")
        description = st.text_area("Description", value=exam.description or "", height=80)
        date = st.date_input("Date", value=exam.date or dt.date.today())
        points_per_question = st.number_input(
            "Default points per question", min_value=0.1, value=float(exam.points_per_question), step=0.5
        )

        col1, col2 = st.columns([1, 1])
        save = col1.form_submit_button("Save changes", type="primary")
        delete = col2.form_submit_button("Delete exam", icon=":material/delete:")

        if save:
            if not name.strip():
                st.error("Exam name is required.")
            else:
                exam_service.update_exam(
                    exam.id,
                    name=name.strip(),
                    course=course.strip() or None,
                    description=description.strip() or None,
                    date=date,
                    points_per_question=points_per_question,
                )
                st.success("Exam updated.")
                st.rerun()

        if delete:
            _confirm_delete_exam(exam.id, exam.name)


def _answer_key_editor(exam_id: int) -> None:
    st.subheader("Answer key")

    existing = exam_service.get_answer_key(exam_id)
    default_size = st.session_state.pop("_pending_answer_key_size", None) or len(existing) or 20

    tab_dropdown, tab_csv = st.tabs(["Dropdowns", "CSV import"])

    with tab_dropdown:
        num_questions = st.number_input(
            "Number of questions", min_value=1, max_value=200, value=int(default_size), key="key_num_questions"
        )
        existing_map = {q.question_number: q.correct_answer for q in existing}

        with st.form("answer_key_form"):
            answers: dict[int, str] = {}
            cols = st.columns(4)
            for i in range(1, int(num_questions) + 1):
                col = cols[(i - 1) % 4]
                default = existing_map.get(i, "A")
                answers[i] = col.selectbox(
                    f"Q{i}", VALID_ANSWERS, index=VALID_ANSWERS.index(default), key=f"answer_key_q{i}"
                )
            points = st.number_input("Points per question", min_value=0.1, value=1.0, step=0.5)
            if st.form_submit_button("Save answer key"):
                exam_service.set_answer_key_from_dropdowns(exam_id, answers, points)
                st.success("Answer key saved.")
                st.rerun()

    with tab_csv:
        st.caption("CSV columns: question,answer,points")
        uploaded = st.file_uploader("Upload answer key CSV", type=["csv"], key="answer_key_csv")
        if uploaded is not None and st.button("Import CSV"):
            warnings = exam_service.set_answer_key_from_csv(exam_id, uploaded.getvalue())
            if warnings:
                st.warning("Some rows were skipped:\n" + "\n".join(warnings))
            st.success("Answer key imported.")
            st.rerun()

    if existing:
        st.dataframe(
            pd.DataFrame([{"question": q.question_number, "answer": q.correct_answer, "points": q.points} for q in existing]),
            hide_index=True,
        )


@st.dialog("Delete submission")
def _confirm_delete_submission(submission_id: int) -> None:
    st.warning(
        f"This permanently deletes submission {submission_id}, its answers, and its images. "
        "This cannot be undone."
    )
    col1, col2 = st.columns(2)
    if col1.button("Delete", type="primary"):
        exam_service.delete_submission(submission_id)
        st.rerun()
    if col2.button("Cancel"):
        st.rerun()


def _submissions_view(exam_id: int) -> None:
    st.subheader("Submissions")
    submissions = exam_service.list_submissions(exam_id)
    if not submissions:
        st.info("No submissions yet for this exam. Use the Upload page to add some.")
        return

    df = pd.DataFrame(
        [
            {
                "id": s.id,
                "status": s.status,
                "score": s.score,
                "percentage": s.percentage,
                "created_at": s.created_at,
            }
            for s in submissions
        ]
    )
    st.dataframe(df, hide_index=True)

    with st.container(horizontal=True):
        submission_id = st.selectbox("Submission to delete", [s.id for s in submissions], label_visibility="collapsed")
        if st.button("Delete submission", icon=":material/delete:"):
            _confirm_delete_submission(submission_id)


def render() -> None:
    st.title("Exams")

    exams = exam_service.list_exams()
    exam_options = {f"{e.name} (id={e.id})": e.id for e in exams}

    _create_exam_form()
    st.divider()

    if not exam_options:
        st.info("No exams yet. Create one above.")
        return

    default_id = st.session_state.get("selected_exam_id")
    labels = list(exam_options.keys())
    default_index = 0
    if default_id is not None:
        for i, (label, eid) in enumerate(exam_options.items()):
            if eid == default_id:
                default_index = i
                break

    selected_label = st.selectbox("Select exam", labels, index=default_index)
    exam_id = exam_options[selected_label]
    st.session_state["selected_exam_id"] = exam_id
    exam = exam_service.get_exam(exam_id)

    _exam_details_editor(exam)
    st.divider()
    _answer_key_editor(exam_id)
    st.divider()
    _submissions_view(exam_id)
