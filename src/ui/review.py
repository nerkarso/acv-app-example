from __future__ import annotations

import streamlit as st

from src.services import exam_service, review_service


def _render_answer_queue(exam_id: int | None) -> None:
    items = review_service.list_review_queue(exam_id)

    if not items:
        st.success("No flagged answers pending review.")
        return

    st.caption(f"{len(items)} answer(s) pending review.")

    for item in items:
        with st.container(border=True):
            cols = st.columns([1, 2, 2])

            with cols[0]:
                if item.crop_image_path:
                    try:
                        st.image(item.crop_image_path, width=150)
                    except Exception:
                        st.caption("Crop image unavailable")

            with cols[1]:
                st.write(f"**Student:** {item.student_number or 'unknown'} - {item.student_name or ''}")
                st.write(f"**Question:** {item.question_number}")
                st.write(f"**Detected:** {item.detected_answer or 'UNKNOWN'} ({item.answer_state})")
                st.write(
                    f"**Confidence:** {item.confidence:.2f}" if item.confidence is not None else "**Confidence:** n/a"
                )
                st.write(f"**Method:** {item.detection_method}")

            with cols[2]:
                choice_cols = st.columns(5)
                labels = ["A", "B", "C", "D", "UNKNOWN"]
                for i, label in enumerate(labels):
                    if choice_cols[i].button(label, key=f"correct_{item.answer_id}_{label}"):
                        review_service.correct_answer(item.answer_id, label)
                        st.rerun()

                if st.button("Confirm as-is", key=f"confirm_{item.answer_id}"):
                    review_service.confirm_answer(item.answer_id)
                    st.rerun()


def _render_submission_issues(exam_id: int | None) -> None:
    issues = review_service.list_submission_issues(exam_id)

    if not issues:
        st.success("No document-detection or student-number issues pending.")
        return

    st.caption(f"{len(issues)} submission(s) with a document/header-level issue.")

    for issue in issues:
        with st.container(border=True):
            cols = st.columns([1, 2, 2])

            with cols[0]:
                image_path = issue.processed_image_path or issue.original_image_path
                try:
                    st.image(image_path, width=180)
                except Exception:
                    st.caption("Image unavailable")

            with cols[1]:
                st.write(f"**Submission:** {issue.submission_id}")
                st.write(f"**Issue:** {issue.error_code}")
                if issue.error_message:
                    st.caption(issue.error_message)
                st.write(f"**Student:** {issue.student_number or 'unknown'} - {issue.student_name or ''}")

            with cols[2]:
                if issue.error_code == "STUDENT_NUMBER_UNREADABLE":
                    with st.form(f"resolve_student_{issue.submission_id}"):
                        student_number = st.text_input("Student number", key=f"sn_{issue.submission_id}")
                        name = st.text_input("Name (optional)", key=f"name_{issue.submission_id}")
                        if st.form_submit_button("Save"):
                            if student_number.strip():
                                review_service.resolve_submission_issue(
                                    issue.submission_id, student_number.strip(), name.strip() or None
                                )
                                st.rerun()
                            else:
                                st.error("Student number is required.")
                else:
                    st.caption(
                        "Document could not be normalized -- re-photograph and re-upload with "
                        "'Force reprocess' on the Upload page."
                    )


def render() -> None:
    st.title("Review Queue")

    exams = exam_service.list_exams()
    exam_options = {"All exams": None} | {f"{e.name}": e.id for e in exams}
    selected_label = st.selectbox("Exam", list(exam_options.keys()))
    exam_id = exam_options[selected_label]

    tab_answers, tab_submissions = st.tabs(["Flagged Answers", "Document / Student Number Issues"])

    with tab_answers:
        _render_answer_queue(exam_id)

    with tab_submissions:
        _render_submission_issues(exam_id)
