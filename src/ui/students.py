from __future__ import annotations

import streamlit as st

from src.services import student_service
from src.ui._answer_editor import render_answers_editor
from src.ui._format import badge
from src.ui._record_table import record_table
from src.ui._table_controls import get_page_size, paginate_slice, pagination_controls


@st.dialog("Create student")
def _create_student_dialog() -> None:
    with st.form("create_student_form", clear_on_submit=True, border=False):
        student_number = st.text_input("Student number")
        name = st.text_input("Name")
        if st.form_submit_button("Create student"):
            if not student_number.strip():
                st.error("Student number is required.")
            else:
                try:
                    student_service.create_student(student_number.strip(), name.strip() or None)
                    st.session_state["_toast"] = "Student created."
                    st.rerun()
                except Exception as exc:
                    st.error(f"Could not create student: {exc}")


@st.dialog("Delete student")
def _confirm_delete_student(student_id: int, student_number: str) -> None:
    st.warning(
        f"This permanently deletes student '{student_number}'. Their existing answer sheets "
        "are kept but unlinked from this student. This cannot be undone."
    )
    with st.container(horizontal=True, horizontal_alignment="distribute"):
        if st.button("Delete", type="primary"):
            student_service.delete_student(student_id)
            st.session_state.pop("selected_student_id", None)
            st.session_state["_student_view"] = "list"
            st.rerun()
        if st.button("Cancel"):
            st.rerun()


@st.dialog("Delete students")
def _confirm_bulk_delete_students(student_ids: list[int]) -> None:
    st.warning(
        f"This permanently deletes {len(student_ids)} student(s). Their existing answer sheets "
        "are kept but unlinked. This cannot be undone."
    )
    with st.container(horizontal=True, horizontal_alignment="distribute"):
        if st.button("Delete", type="primary"):
            for student_id in student_ids:
                student_service.delete_student(student_id)
            if st.session_state.get("selected_student_id") in student_ids:
                st.session_state.pop("selected_student_id", None)
                st.session_state["_student_view"] = "list"
            st.rerun()
        if st.button("Cancel"):
            st.rerun()


def _edit_student(student) -> None:
    with st.form("edit_student_form"):
        student_number = st.text_input("Student number", value=student.student_number)
        name = st.text_input("Name", value=student.name or "")

        with st.container(horizontal=True, horizontal_alignment="distribute"):
            save = st.form_submit_button("Save changes", type="primary")
            delete = st.form_submit_button("Delete student", icon=":material/delete:")

        if save:
            if not (student_number or "").strip():
                st.error("Student number is required.")
            else:
                try:
                    student_service.update_student(
                        student.id, (student_number or "").strip(), (name or "").strip() or None
                    )
                    st.toast("Student updated.", icon=":material/check_circle:")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Could not update student: {exc}")

        if delete:
            _confirm_delete_student(student.id, student.student_number)


def _submissions_view(student_id: int) -> None:
    submissions = student_service.list_submissions_for_student(student_id)
    if not submissions:
        st.info("No answer sheets yet for this student.")
        return

    options = {
        f"{s.exam_name} - {s.created_at:%Y-%m-%d %H:%M} (answer sheet {s.submission_id})": s
        for s in submissions
    }
    chosen = st.selectbox("Answer sheet", list(options.keys()), label_visibility="collapsed")
    submission = options[chosen]

    cols = st.columns(4)
    with cols[0]:
        st.caption("Status")
        badge(submission.status)
    cols[1].metric("Score", submission.score if submission.score is not None else "--")
    cols[2].metric("Total points", submission.total_points if submission.total_points is not None else "--")
    cols[3].metric(
        "Percentage", f"{submission.percentage:.1f}%" if submission.percentage is not None else "--"
    )

    image_path = submission.processed_image_path or submission.original_image_path
    try:
        st.image(image_path, width="stretch")
    except Exception:
        st.caption("Image unavailable.")

    st.subheader("Answers")
    render_answers_editor(submission.submission_id, key_prefix="student_answers")


def _render_student_detail(student) -> None:
    if st.button("Back to students", icon=":material/arrow_back:"):
        st.session_state["_student_view"] = "list"
        st.rerun()

    st.header(student.name or student.student_number)

    tab_details, tab_submissions = st.tabs(["Student details", "Answer sheets"])
    with tab_details:
        _edit_student(student)
    with tab_submissions:
        _submissions_view(student.id)


def _render_student_list(students) -> None:
    if st.button("Create student", icon=":material/add:"):
        _create_student_dialog()

    if not students:
        st.info("No students yet. Students are also created automatically when an answer sheet's "
                 "header is read successfully, or created manually above.")
        return

    search = st.text_input(
        "Search students", key="students_search", placeholder="Search by student number or name",
        label_visibility="collapsed",
    )
    if search:
        needle = search.lower()
        students = [
            s for s in students
            if needle in (s.student_number or "").lower() or needle in (s.name or "").lower()
        ]

    sort_state = st.session_state.setdefault("students_sort", {"key": "id", "dir": "asc"})
    sort_key_fns = {
        "id": lambda s: s.id,
        "student_number": lambda s: (s.student_number or "").lower(),
        "name": lambda s: (s.name or "").lower(),
    }
    students = sorted(
        students, key=sort_key_fns[sort_state["key"]], reverse=sort_state["dir"] == "desc"
    )

    if not students:
        st.info("No students match your search.")
        return

    prior_selected = st.session_state.get("students_table", {}).get("selected", [])
    page_size = get_page_size("students_page")
    page_students, num_pages, total, page = paginate_slice(
        students, key="students_page", page_size=page_size
    )
    rows = [
        {"id": s.id, "student_number": s.student_number, "name": s.name or ""}
        for s in page_students
    ]
    columns = [
        {"key": "id", "label": "ID", "width": 60},
        {"key": "student_number", "label": "Student number", "width": 150},
        {"key": "name", "label": "Name", "width": 220},
    ]

    result = record_table(rows, columns, key="students_table", sort=sort_state)
    if result.sort_requested:
        if result.sort_requested == sort_state["key"]:
            sort_state["dir"] = "asc" if sort_state["dir"] == "desc" else "desc"
        else:
            sort_state["key"] = result.sort_requested
            sort_state["dir"] = "asc"
        st.session_state["students_sort"] = sort_state
        st.session_state["students_page"] = 1
        st.rerun()
    if result.opened:
        st.session_state["selected_student_id"] = result.opened
        st.session_state["_student_view"] = "detail"
        st.rerun()

    left, right = st.columns([1, 1])
    with left:
        if prior_selected and st.button(
            f"Delete selected ({len(prior_selected)})", icon=":material/delete:", type="primary"
        ):
            _confirm_bulk_delete_students(prior_selected)
    with right:
        pagination_controls(num_pages, page, page_size, total, key="students_page")


def render() -> None:
    st.title("Students")

    toast_message = st.session_state.pop("_toast", None)
    if toast_message:
        st.toast(toast_message, icon=":material/check_circle:")

    students = student_service.list_students()

    student_id = st.session_state.get("selected_student_id")
    if st.session_state.get("_student_view") == "detail" and student_id is not None:
        student = student_service.get_student(student_id)
        if student is None:
            st.session_state["_student_view"] = "list"
            st.rerun()
        _render_student_detail(student)
        return

    _render_student_list(students)
