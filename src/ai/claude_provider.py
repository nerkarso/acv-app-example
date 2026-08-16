"""Default cloud escalation provider: Claude vision API.

Sends only the small answer/field crop, never the full page. The prompt
constrains output to the AnswerDetection JSON shape and 4-state domain.
"""

from __future__ import annotations

import base64
import json
import logging

import cv2
import numpy as np

from src.ai.base import VisionProvider
from src.config import settings
from src.schemas import AnswerDetection, AnswerState, DetectionMethod

logger = logging.getLogger(__name__)

CLAUDE_MODEL = "claude-opus-4-6"

_SYSTEM_PROMPT = (
    "You are grading a handwritten multiple-choice answer crop from a scanned "
    "exam sheet. Look only at the image provided. Respond with a single JSON "
    'object matching exactly this shape: {"answer": "A"|"B"|"C"|"D"|null, '
    '"state": "clear"|"struck_through"|"blank"|"ambiguous", "confidence": '
    "0.0-1.0}. Never guess a letter you are not reasonably confident about -- "
    'use null with state "ambiguous" or "blank" instead. If the letter has a '
    'line drawn through it, use state "struck_through". Respond with only the '
    "JSON object, no other text."
)


class ClaudeVisionProvider(VisionProvider):
    def __init__(self) -> None:
        self._client = None

    def _get_client(self):
        if self._client is None:
            from anthropic import Anthropic

            self._client = Anthropic(api_key=settings.anthropic_api_key)
        return self._client

    @property
    def is_available(self) -> bool:
        return bool(settings.anthropic_api_key)

    def classify_answer(self, image: np.ndarray, valid_answers: list[str]) -> AnswerDetection:
        if not self.is_available:
            return AnswerDetection(
                answer=None,
                state=AnswerState.AMBIGUOUS,
                confidence=0.0,
                method=DetectionMethod.CLOUD_VLM,
            )

        success, buffer = cv2.imencode(".png", image)
        if not success:
            logger.error("Failed to encode crop for Claude escalation")
            return AnswerDetection(
                answer=None,
                state=AnswerState.AMBIGUOUS,
                confidence=0.0,
                method=DetectionMethod.CLOUD_VLM,
            )
        image_b64 = base64.b64encode(buffer.tobytes()).decode("utf-8")

        try:
            client = self._get_client()
            message = client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=256,
                system=_SYSTEM_PROMPT,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": image_b64,
                                },
                            },
                            {
                                "type": "text",
                                "text": f"Valid answers: {valid_answers}. Classify this crop.",
                            },
                        ],
                    }
                ],
            )
            text_blocks = [block.text for block in message.content if block.type == "text"]
            raw_text = "".join(text_blocks).strip()
            parsed = json.loads(raw_text)

            answer = parsed.get("answer")
            if answer not in valid_answers:
                answer = None
            state = parsed.get("state", "ambiguous")
            if state not in {s.value for s in AnswerState}:
                state = "ambiguous"
            confidence = float(parsed.get("confidence", 0.0))
            confidence = max(0.0, min(1.0, confidence))

            return AnswerDetection(
                answer=answer,
                state=AnswerState(state),
                confidence=confidence,
                method=DetectionMethod.CLOUD_VLM,
            )
        except Exception:
            logger.exception("Claude vision escalation failed")
            return AnswerDetection(
                answer=None,
                state=AnswerState.AMBIGUOUS,
                confidence=0.0,
                method=DetectionMethod.CLOUD_VLM,
            )
