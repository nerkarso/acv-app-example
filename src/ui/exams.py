from __future__ import annotations

import datetime as dt

import pandas as pd
import streamlit as st

from src.schemas import VALID_ANSWERS
from src.services import exam_service
from src.ui._record_table import record_table
from src.ui._table_controls import get_page_size, paginate_slice, pagination_controls


@st.dialog("Create exam")
def _create_exam_dialog() -> None:
    with st.form("create_exam_form", clear_on_submit=True, border=False):
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
            st.session_state["_exam_view"] = "detail"
            st.session_state["_toast"] = f"Exam '{name}' created. Define the answer key below."
            st.rerun()


@st.dialog("Delete exam")
def _confirm_delete_exam(exam_id: int, exam_name: str) -> None:
    st.warning(
        f"This permanently deletes '{exam_name}' and all of its questions, submissions, "
        "answers, and images. This cannot be undone."
    )
    with st.container(horizontal=True, horizontal_alignment="distribute"):
        if st.button("Delete", type="primary"):
            exam_service.delete_exam(exam_id)
            st.session_state.pop("selected_exam_id", None)
            st.session_state["_exam_view"] = "list"
            st.rerun()
        if st.button("Cancel"):
            st.rerun()


@st.dialog("Delete exams")
def _confirm_bulk_delete_exams(exam_ids: list[int]) -> None:
    st.warning(
        f"This permanently deletes {len(exam_ids)} exam(s) and all of their questions, submissions, "
        "answers, and images. This cannot be undone."
    )
    with st.container(horizontal=True, horizontal_alignment="distribute"):
        if st.button("Delete", type="primary"):
            for exam_id in exam_ids:
                exam_service.delete_exam(exam_id)
            if st.session_state.get("selected_exam_id") in exam_ids:
                st.session_state.pop("selected_exam_id", None)
                st.session_state["_exam_view"] = "list"
            st.rerun()
        if st.button("Cancel"):
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

        with st.container(horizontal=True, horizontal_alignment="distribute"):
            save = st.form_submit_button("Save changes", type="primary")
            delete = st.form_submit_button("Delete exam", icon=":material/delete:")

        if save:
            if not (name or "").strip():
                st.error("Exam name is required.")
            else:
                exam_service.update_exam(
                    exam.id,
                    name=(name or "").strip(),
                    course=(course or "").strip() or None,
                    description=(description or "").strip() or None,
                    date=date,
                    points_per_question=points_per_question,
                )
                st.toast("Exam updated.", icon=":material/check_circle:")
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
                st.toast("Answer key saved.", icon=":material/check_circle:")
                st.rerun()

    with tab_csv:
        st.caption("CSV columns: question,answer,points")
        uploaded = st.file_uploader("Upload answer key CSV", type=["csv"], key="answer_key_csv")
        if uploaded is not None and st.button("Import CSV"):
            warnings = exam_service.set_answer_key_from_csv(exam_id, uploaded.getvalue())
            if warnings:
                st.warning("Some rows were skipped:\n" + "\n".join(warnings))
            st.toast("Answer key imported.", icon=":material/check_circle:")
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
    with st.container(horizontal=True, horizontal_alignment="distribute"):
        if st.button("Delete", type="primary"):
            exam_service.delete_submission(submission_id)
            st.rerun()
        if st.button("Cancel"):
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


def _render_exam_detail(exam) -> None:
    if st.button("Back to exams", icon=":material/arrow_back:"):
        st.session_state["_exam_view"] = "list"
        st.rerun()

    st.header(exam.name)
    _exam_details_editor(exam)
    st.divider()
    _answer_key_editor(exam.id)
    st.divider()
    _submissions_view(exam.id)


def _render_exam_list(exams) -> None:
    if st.button("Create exam", icon=":material/add:"):
        _create_exam_dialog()

    if not exams:
        st.info("No exams yet. Create one above.")
        return

    search = st.text_input(
        "Search exams", key="exams_search", placeholder="Search by name or course",
        label_visibility="collapsed",
    )
    if search:
        needle = search.lower()
        exams = [
            e for e in exams
            if needle in (e.name or "").lower() or needle in (e.course or "").lower()
        ]

    sort_state = st.session_state.setdefault("exams_sort", {"key": "id", "dir": "desc"})
    sort_key_fns = {
        "id": lambda e: e.id,
        "name": lambda e: (e.name or "").lower(),
        "course": lambda e: (e.course or "").lower(),
        "date": lambda e: e.date or dt.date.min,
        "points": lambda e: e.points_per_question,
    }
    exams = sorted(exams, key=sort_key_fns[sort_state["key"]], reverse=sort_state["dir"] == "desc")

    if not exams:
        st.info("No exams match your search.")
        return

    prior_selected = st.session_state.get("exams_table", {}).get("selected", [])
    page_size = get_page_size("exams_page")
    page_exams, num_pages, total, page = paginate_slice(exams, key="exams_page", page_size=page_size)
    rows = [
        {
            "id": e.id,
            "name": e.name,
            "course": e.course or "",
            "date": str(e.date) if e.date else "",
            "points": e.points_per_question,
        }
        for e in page_exams
    ]
    columns = [
        {"key": "id", "label": "ID", "width": 60},
        {"key": "name", "label": "Name", "width": 220},
        {"key": "course", "label": "Course", "width": 140},
        {"key": "date", "label": "Date", "width": 110},
        {"key": "points", "label": "Points/question", "width": 130},
    ]

    result = record_table(rows, columns, key="exams_table", sort=sort_state)
    if result.sort_requested:
        if result.sort_requested == sort_state["key"]:
            sort_state["dir"] = "asc" if sort_state["dir"] == "desc" else "desc"
        else:
            sort_state["key"] = result.sort_requested
            sort_state["dir"] = "asc"
        st.session_state["exams_sort"] = sort_state
        st.session_state["exams_page"] = 1
        st.rerun()
    if result.opened:
        st.session_state["selected_exam_id"] = result.opened
        st.session_state["_exam_view"] = "detail"
        st.rerun()

    left, right = st.columns([1, 1])
    with left:
        if prior_selected and st.button(
            f"Delete selected ({len(prior_selected)})", icon=":material/delete:", type="primary"
        ):
            _confirm_bulk_delete_exams(prior_selected)
    with right:
        pagination_controls(num_pages, page, page_size, total, key="exams_page")


def render() -> None:
    st.title("Exams")

    toast_message = st.session_state.pop("_toast", None)
    if toast_message:
        st.toast(toast_message, icon=":material/check_circle:")

    exams = exam_service.list_exams()

    exam_id = st.session_state.get("selected_exam_id")
    if st.session_state.get("_exam_view") == "detail" and exam_id is not None:
        exam = exam_service.get_exam(exam_id)
        if exam is None:
            st.session_state["_exam_view"] = "list"
            st.rerun()
        _render_exam_detail(exam)
        return

    _render_exam_list(exams)
