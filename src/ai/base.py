"""Pluggable vision-provider interface for low-confidence escalation.

Pipeline code depends only on this interface, never on a concrete provider,
so Claude/local-VLM/none are interchangeable via VISION_PROVIDER.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from src.schemas import AnswerDetection


class VisionProvider(ABC):
    @abstractmethod
    def classify_answer(self, image: np.ndarray, valid_answers: list[str]) -> AnswerDetection:
        """Classify a single small crop (one answer or header field) into one
        of `valid_answers`, or None with an appropriate state -- never guessed."""
        raise NotImplementedError

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Whether this provider is configured and ready to be called."""
        raise NotImplementedError
