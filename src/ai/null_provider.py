"""VISION_PROVIDER=none -- escalation disabled entirely.

The application must remain fully functional with this provider active:
anything that would have escalated instead goes straight to manual review.
"""

from __future__ import annotations

import numpy as np

from src.ai.base import VisionProvider
from src.schemas import SubmissionEscalationResult


class NullVisionProvider(VisionProvider):
    def read_submission(
        self, page_image: np.ndarray, field_names: list[str], question_numbers: list[int]
    ) -> SubmissionEscalationResult:
        return SubmissionEscalationResult(fields={}, answers={})

    @property
    def is_available(self) -> bool:
        return False
