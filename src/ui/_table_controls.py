from __future__ import annotations

from typing import TypeVar

import streamlit as st

T = TypeVar("T")

PAGE_SIZE_OPTIONS = (10, 25, 50, 100)


def get_page_size(key: str, *, default: int = 10) -> int:
    return int(st.session_state.get(f"{key}_size", default))


def paginate_slice(
    items: list[T], *, key: str, page_size: int = 10
) -> tuple[list[T], int, int, int]:
    total = len(items)

    prev_size_key = f"_{key}_prev_size"
    if st.session_state.get(prev_size_key) not in (None, page_size):
        st.session_state[key] = 1
    st.session_state[prev_size_key] = page_size

    num_pages = max(1, -(-total // page_size))
    if key in st.session_state:
        st.session_state[key] = min(max(int(st.session_state[key]), 1), num_pages)
    page = st.session_state.get(key, 1)

    start = (page - 1) * page_size
    return items[start : start + page_size], num_pages, total, page


def pagination_controls(
    num_pages: int, page: int, page_size: int, total: int, *, key: str
) -> None:
    with st.container(horizontal=True, horizontal_alignment="right", vertical_alignment="center"):
        if total:
            start = (page - 1) * page_size + 1
            end = min(page * page_size, total)
            st.caption(f"Showing {start}-{end} of {total}")
        if num_pages > 1:
            st.pagination(num_pages, key=key)
        st.selectbox(
            "Rows per page",
            PAGE_SIZE_OPTIONS,
            index=PAGE_SIZE_OPTIONS.index(page_size),
            key=f"{key}_size",
            label_visibility="collapsed",
            width=80,
        )
