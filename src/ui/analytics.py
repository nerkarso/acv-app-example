from __future__ import annotations

import plotly.express as px
import streamlit as st

from src.services import analytics_service, exam_service


def render() -> None:
    st.title("Analytics")

    exams = exam_service.list_exams()
    if not exams:
        st.info("No exams yet.")
        return

    exam_options = {f"{e.name}": e.id for e in exams}
    selected_label = st.selectbox("Exam", list(exam_options.keys()))
    exam_id = exam_options[selected_label]
    exam = exam_service.get_exam(exam_id)

    st.subheader("Grade Distribution")
    distribution = analytics_service.get_grade_distribution(exam_id)
    if distribution["percentages"]:
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Mean", f"{distribution['mean']:.1f}%")
        col2.metric("Median", f"{distribution['median']:.1f}%")
        col3.metric("Min", f"{distribution['min']:.1f}%")
        col4.metric("Max", f"{distribution['max']:.1f}%")
        col5.metric("Std Dev", f"{distribution['stdev']:.1f}")
        st.caption(f"Pass rate: {distribution['pass_rate']:.1f}%")

        fig = px.histogram(x=distribution["percentages"], nbins=20, labels={"x": "Percentage"})
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("No graded submissions yet.")

    st.divider()
    st.subheader("Per-Question Difficulty (easiest -> hardest)")
    difficulty_df = analytics_service.get_question_difficulty(exam_id)
    if not difficulty_df.empty:
        fig = px.bar(difficulty_df, x="question_number", y="pct_correct", labels={"pct_correct": "% Correct"})
        st.plotly_chart(fig, width="stretch")
        st.dataframe(difficulty_df, hide_index=True)
    else:
        st.info("No answer data yet.")

    st.divider()
    st.subheader("Trend Across Exams (same course)")
    if exam and exam.course:
        trend_df = analytics_service.get_course_trend(exam.course)
        if not trend_df.empty:
            fig = px.line(trend_df, x="exam_name", y="average_pct", markers=True)
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("No other exams in this course yet.")
    else:
        st.info("This exam has no course set.")

    st.divider()
    st.subheader("OCR vs. Escalation Usage (cost visibility)")
    breakdown_df = analytics_service.get_ocr_vs_escalation_breakdown(exam_id)
    if not breakdown_df.empty:
        fig = px.pie(breakdown_df, names="detection_method", values="count")
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("No detections yet.")

    st.divider()
    st.subheader("Export")
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "Download results CSV",
            data=analytics_service.export_results_csv(exam_id),
            file_name=f"exam_{exam_id}_results.csv",
            mime="text/csv",
        )
        st.download_button(
            "Download results XLSX",
            data=analytics_service.export_results_xlsx(exam_id),
            file_name=f"exam_{exam_id}_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    with col2:
        st.download_button(
            "Download question analysis CSV",
            data=analytics_service.export_question_analysis_csv(exam_id),
            file_name=f"exam_{exam_id}_question_analysis.csv",
            mime="text/csv",
        )
        st.download_button(
            "Download question analysis XLSX",
            data=analytics_service.export_question_analysis_xlsx(exam_id),
            file_name=f"exam_{exam_id}_question_analysis.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
