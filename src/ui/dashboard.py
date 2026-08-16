from __future__ import annotations

import streamlit as st

from src.services.analytics_service import get_dashboard_stats


def render() -> None:
    st.title("Dashboard")

    stats = get_dashboard_stats()

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Exams", stats.total_exams)
    col2.metric("Total Students", stats.total_students)
    col3.metric(
        "Average Score",
        f"{stats.average_score_pct:.1f}%" if stats.average_score_pct is not None else "-",
    )
    col4.metric(
        "Pass Rate",
        f"{stats.pass_rate_pct:.1f}%" if stats.pass_rate_pct is not None else "-",
    )
    col5.metric("Exams Requiring Review", stats.exams_requiring_review)

    st.divider()
    st.caption(
        "Use the Exams page to create an exam and answer key, Upload to batch-process "
        "papers, Review to resolve flagged answers, and Analytics/Results for reporting."
    )
