import logging

import streamlit as st

from src.config import settings
from src.database.database import init_db
from src.ui import analytics, dashboard, exams, review, results, students, upload

st.set_page_config(page_title="Exam Grader", layout="wide")

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

init_db()

pages = {
    "Overview": [
        st.Page(dashboard.render, title="Dashboard", icon=":material/dashboard:", url_path="dashboard", default=True),
    ],
    "Grading": [
        st.Page(upload.render, title="Upload", icon=":material/upload_file:", url_path="upload"),
        st.Page(review.render, title="Review", icon=":material/fact_check:", url_path="review"),
    ],
    "Insights": [
        st.Page(results.render, title="Results", icon=":material/table_chart:", url_path="results"),
        st.Page(analytics.render, title="Analytics", icon=":material/bar_chart:", url_path="analytics"),
    ],
    "Manage": [
        st.Page(exams.render, title="Exams", icon=":material/quiz:", url_path="exams"),
        st.Page(students.render, title="Students", icon=":material/group:", url_path="students"),
    ],
}

navigation = st.navigation(pages)
navigation.run()
