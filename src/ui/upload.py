from __future__ import annotations

import pandas as pd
import streamlit as st

from src.services import exam_service, processing_service
from src.ui._format import humanize


def render() -> None:
    st.title("Upload & Process")

    exams = exam_service.list_exams()
    if not exams:
        st.info("Create an exam first on the Exams page.")
        return

    exam_options = {f"{e.name}": e.id for e in exams}
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
            total = len(submission_ids)
            tally = None

            with st.status(f"Processing 0/{total} paper(s)...", expanded=True) as status:
                progress_bar = st.progress(0.0)
                current_file = st.empty()
                metric_cols = st.columns(5)
                metric_successful = metric_cols[0].empty()
                metric_needs_review = metric_cols[1].empty()
                metric_failed = metric_cols[2].empty()
                metric_ocr_accepted = metric_cols[3].empty()
                metric_escalated = metric_cols[4].empty()

                for tally, outcome in processing_service.process_batch(exam_id, submission_ids):
                    progress_bar.progress(tally.processed / total)
                    if outcome is not None:
                        current_file.caption(f"Last processed: **{outcome.file_name}** -- {humanize(outcome.status)}")
                    metric_successful.metric("Successful", tally.successful)
                    metric_needs_review.metric("Needs review", tally.needs_review)
                    metric_failed.metric("Failed", tally.failed)
                    metric_ocr_accepted.metric("OCR accepted", tally.ocr_accepted_count)
                    metric_escalated.metric("Escalated", tally.escalated_count)
                    status.update(label=f"Processing {tally.processed}/{total} paper(s)...")

                status.update(
                    label=f"Processed {total} paper(s)",
                    state="error" if tally and tally.failed else "complete",
                )

            if tally is not None and tally.outcomes:
                st.subheader("Results")
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "file_name": o.file_name,
                                "status": humanize(o.status),
                                "error_code": humanize(o.error_code),
                                "ocr_accepted": o.ocr_accepted_count,
                                "escalated": o.escalated_count,
                                "manual_review": o.manual_review_count,
                            }
                            for o in tally.outcomes
                        ]
                    ),
                    hide_index=True,
                    column_config={
                        "file_name": st.column_config.TextColumn("File"),
                        "status": st.column_config.TextColumn("Status"),
                        "error_code": st.column_config.TextColumn("Error"),
                        "ocr_accepted": st.column_config.NumberColumn("OCR accepted"),
                        "escalated": st.column_config.NumberColumn("Escalated"),
                        "manual_review": st.column_config.NumberColumn("Manual review"),
                    },
                )

            st.success("Batch processing complete.")
