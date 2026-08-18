"""Pluggable vision-provider interface for low-confidence escalation.

Pipeline code depends only on this interface, never on a concrete provider,
so Claude/none are interchangeable via VISION_PROVIDER.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from src.schemas import SubmissionEscalationResult


class VisionProvider(ABC):
    @abstractmethod
    def read_submission(
        self, page_image: np.ndarray, field_names: list[str], question_numbers: list[int]
    ) -> SubmissionEscalationResult:
        """Read every listed header field and classify every listed answer
        from a single page image in one call -- the given field/question
        lists are exactly what PaddleOCR could not confidently resolve on
        its own. Never guess a value that isn't reasonably legible; leave it
        absent from the result (or null) instead."""
        raise NotImplementedError

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Whether this provider is configured and ready to be called."""
        raise NotImplementedError
