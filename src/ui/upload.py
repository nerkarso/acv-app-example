from __future__ import annotations

import streamlit as st

from src.services import exam_service, processing_service


def render() -> None:
    st.title("Upload & Process")

    exams = exam_service.list_exams()
    if not exams:
        st.info("Create an exam first on the Exams page.")
        return

    exam_options = {f"{e.name} (id={e.id})": e.id for e in exams}
    selected_label = st.selectbox("Exam", list(exam_options.keys()))
    exam_id = exam_options[selected_label]

    exam = exam_service.get_exam(exam_id)
    if not exam or exam.total_questions == 0:
        st.warning("This exam has no answer key yet. Define one on the Exams page before grading.")

    files = st.file_uploader(
        "Exam paper images", type=["jpg", "jpeg", "png"], accept_multiple_files=True
    )

    force_reprocess = st.checkbox("Force reprocess duplicates (skip duplicate check)", value=False)

    if files and st.button("Start Processing", type="primary"):
        submission_ids: list[int] = []
        skipped: list[str] = []

        for f in files:
            file_bytes = f.getvalue()
            file_hash = processing_service.compute_file_hash(file_bytes)

            if not force_reprocess:
                dup_id = processing_service.find_duplicate(exam_id, file_hash)
                if dup_id is not None:
                    skipped.append(f.name)
                    continue

            submission_id = processing_service.create_pending_submission(exam_id, file_bytes, f.name)
            submission_ids.append(submission_id)

        if skipped:
            st.warning(f"Skipped {len(skipped)} duplicate file(s): {', '.join(skipped)}")

        if submission_ids:
            progress_bar = st.progress(0.0)
            status_placeholder = st.empty()
            tally_placeholder = st.empty()

            total = len(submission_ids)
            for tally, outcome in processing_service.process_batch(exam_id, submission_ids):
                progress_bar.progress(tally.processed / total)
                status_placeholder.write(
                    f"Processed {tally.processed}/{total} -- last: {outcome.file_name} ({outcome.status})"
                )
                tally_placeholder.write(
                    f"Successful: {tally.successful} | Needs review: {tally.needs_review} | "
                    f"Failed: {tally.failed} | OCR accepted: {tally.ocr_accepted_count} | "
                    f"Escalated to vision provider: {tally.escalated_count}"
                )

            st.success("Batch processing complete.")
