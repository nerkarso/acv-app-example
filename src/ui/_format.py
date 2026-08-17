from __future__ import annotations

from typing import Literal

import streamlit as st

BadgeColor = Literal["red", "orange", "yellow", "blue", "green", "violet", "gray", "grey", "primary"]

_BADGE_COLORS: dict[str, BadgeColor] = {
    # submission status
    "pending": "gray",
    "processing": "blue",
    "completed": "green",
    "needs_review": "orange",
    "failed": "red",
    # answer state
    "clear": "green",
    "struck_through": "orange",
    "blank": "gray",
    "ambiguous": "orange",
    # review status
    "not_required": "gray",
    "confirmed": "green",
    "corrected": "blue",
}


def humanize(value: str | None) -> str:
    """Turn a raw snake_case enum value (e.g. "needs_review") into a
    human-readable label (e.g. "Needs review") for display in the UI."""
    if not value:
        return ""
    return value.replace("_", " ").capitalize()


def badge(value: str | None, *, default_color: BadgeColor = "gray") -> None:
    """Render a raw enum value (status, answer state, review status, or
    error code) as a colored st.badge pill instead of plain text."""
    if not value:
        return
    color = _BADGE_COLORS.get(value.lower(), default_color)
    st.badge(humanize(value), color=color)
