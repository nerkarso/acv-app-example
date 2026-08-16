"""VISION_PROVIDER=none -- escalation disabled entirely.

The application must remain fully functional with this provider active:
anything that would have escalated instead goes straight to manual review.
"""

from __future__ import annotations

import numpy as np

from src.ai.base import VisionProvider
from src.schemas import AnswerDetection, AnswerState, DetectionMethod


class NullVisionProvider(VisionProvider):
    def classify_answer(self, image: np.ndarray, valid_answers: list[str]) -> AnswerDetection:
        return AnswerDetection(
            answer=None,
            state=AnswerState.AMBIGUOUS,
            confidence=0.0,
            method=DetectionMethod.CLOUD_VLM,
        )

    @property
    def is_available(self) -> bool:
        return False
