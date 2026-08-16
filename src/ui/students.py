from __future__ import annotations

import pandas as pd
import streamlit as st

from src.services import student_service


def _create_student_form() -> None:
    with st.expander("Add student", icon=":material/person_add:"):
        with st.form("create_student_form"):
            student_number = st.text_input("Student number")
            name = st.text_input("Name")
            if st.form_submit_button("Add student"):
                if not student_number.strip():
                    st.error("Student number is required.")
                else:
                    try:
                        student_service.create_student(student_number.strip(), name.strip() or None)
                        st.success("Student added.")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Could not add student: {exc}")


@st.dialog("Delete student")
def _confirm_delete_student(student_id: int, student_number: str) -> None:
    st.warning(
        f"This permanently deletes student '{student_number}'. Their existing submissions "
        "are kept but unlinked from this student. This cannot be undone."
    )
    col1, col2 = st.columns(2)
    if col1.button("Delete", type="primary"):
        student_service.delete_student(student_id)
        st.rerun()
    if col2.button("Cancel"):
        st.rerun()


def _edit_student(student) -> None:
    st.subheader("Edit student")
    with st.form("edit_student_form"):
        student_number = st.text_input("Student number", value=student.student_number)
        name = st.text_input("Name", value=student.name or "")

        col1, col2 = st.columns(2)
        save = col1.form_submit_button("Save changes", type="primary")
        delete = col2.form_submit_button("Delete student", icon=":material/delete:")

        if save:
            if not student_number.strip():
                st.error("Student number is required.")
            else:
                try:
                    student_service.update_student(student.id, student_number.strip(), name.strip() or None)
                    st.success("Student updated.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Could not update student: {exc}")

        if delete:
            _confirm_delete_student(student.id, student.student_number)


def render() -> None:
    st.title("Students")

    _create_student_form()
    st.divider()

    students = student_service.list_students()
    if not students:
        st.info("No students yet. Students are also created automatically when a submission's "
                 "header is read successfully, or added manually above.")
        return

    st.dataframe(
        pd.DataFrame(
            [{"id": s.id, "student_number": s.student_number, "name": s.name} for s in students]
        ),
        hide_index=True,
    )

    st.divider()
    options = {f"{s.student_number} - {s.name or '(no name)'} (id={s.id})": s.id for s in students}
    selected_label = st.selectbox("Select student to edit", list(options.keys()))
    student = student_service.get_student(options[selected_label])
    if student is not None:
        _edit_student(student)
