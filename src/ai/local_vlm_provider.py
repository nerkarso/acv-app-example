"""Optional local VLM escalation provider (e.g. a Qwen-VL-compatible server).

Drop-in alternative to ClaudeVisionProvider -- same VisionProvider interface,
no API key/cost, requires a locally hosted model reachable via HTTP.
Expects an OpenAI-chat-completions-style endpoint that accepts an image and
returns a JSON object matching the AnswerDetection shape.
"""

from __future__ import annotations

import base64
import json
import logging

import cv2
import numpy as np
import urllib.request

from src.ai.base import VisionProvider
from src.config import settings
from src.schemas import AnswerDetection, AnswerState, DetectionMethod

logger = logging.getLogger(__name__)

_PROMPT = (
    "Classify this handwritten multiple-choice answer crop. Respond with only "
    'a JSON object: {"answer": "A"|"B"|"C"|"D"|null, "state": '
    '"clear"|"struck_through"|"blank"|"ambiguous", "confidence": 0.0-1.0}. '
    "Never guess an answer you are not confident about."
)


class LocalVLMProvider(VisionProvider):
    def __init__(self, endpoint: str | None = None, timeout: float = 30.0) -> None:
        self.endpoint = endpoint or settings.local_vlm_endpoint
        self.timeout = timeout

    @property
    def is_available(self) -> bool:
        return bool(self.endpoint)

    def classify_answer(self, image: np.ndarray, valid_answers: list[str]) -> AnswerDetection:
        if not self.is_available:
            return AnswerDetection(
                answer=None,
                state=AnswerState.AMBIGUOUS,
                confidence=0.0,
                method=DetectionMethod.LOCAL_VLM,
            )

        success, buffer = cv2.imencode(".png", image)
        if not success:
            logger.error("Failed to encode crop for local VLM escalation")
            return AnswerDetection(
                answer=None,
                state=AnswerState.AMBIGUOUS,
                confidence=0.0,
                method=DetectionMethod.LOCAL_VLM,
            )
        image_b64 = base64.b64encode(buffer.tobytes()).decode("utf-8")

        payload = {
            "prompt": f"{_PROMPT} Valid answers: {valid_answers}.",
            "image_base64": image_b64,
        }

        try:
            request = urllib.request.Request(
                url=f"{self.endpoint.rstrip('/')}/classify",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                parsed = json.loads(response.read().decode("utf-8"))

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
                method=DetectionMethod.LOCAL_VLM,
            )
        except Exception:
            logger.exception("Local VLM escalation failed")
            return AnswerDetection(
                answer=None,
                state=AnswerState.AMBIGUOUS,
                confidence=0.0,
                method=DetectionMethod.LOCAL_VLM,
            )
