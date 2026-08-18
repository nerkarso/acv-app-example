"""Default cloud escalation provider: Claude vision API.

One call per submission: the full page image plus the exact list of header
fields and question numbers PaddleOCR left uncertain, all read back in a
single JSON response. This replaces one API call per uncertain item (which
adds up fast on a multi-page batch and eats into rate limits) with at most
one call per submission that has anything left uncertain, while also giving
the model the surrounding page context instead of an isolated crop.
"""

from __future__ import annotations

import base64
import json
import logging
import re

import cv2
import numpy as np

from src.ai.base import VisionProvider
from src.config import settings
from src.schemas import (
    VALID_ANSWERS,
    AnswerDetection,
    AnswerState,
    DetectionMethod,
    OCRFieldResult,
    SubmissionEscalationResult,
)

logger = logging.getLogger(__name__)

JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)

SUBMISSION_SYSTEM_PROMPT = (
    "You are grading a scanned exam answer sheet. You are given one image "
    "of the full page. Respond with a single JSON object matching exactly "
    'this shape: {"fields": {"<field_name>": {"text": <string or null>, '
    '"confidence": <0.0-1.0>}, ...}, "answers": {"<question_number>": '
    '{"answer": "A"|"B"|"C"|"D"|null, "state": "clear"|"struck_through"|'
    '"blank"|"ambiguous", "confidence": <0.0-1.0>}, ...}}. Only include the '
    "exact field names and question numbers listed in the request -- do "
    "not add, rename, or omit any of them. Transcribe field values exactly "
    "as written, correcting only obvious character-recognition mistakes. "
    "For answers, never guess a letter you are not reasonably confident "
    'about -- use null with state "ambiguous" or "blank" instead. If an '
    'answer letter has a line drawn through it, use state "struck_through". '
    "Respond with only the JSON object, no other text."
)

FIELD_HINTS = {
    "student_number": "student number, format SE/00YY/000, e.g. SE/0023/001",
    "name": "student's full name",
}


def _extract_json_object(raw_text: str) -> dict:
    """Parse a JSON object from a model response, tolerating markdown code
    fences (e.g. ```json ... ```) that some proxied/local models add despite
    being told to respond with only JSON."""
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        match = JSON_OBJECT_RE.search(raw_text)
        if not match:
            raise
        return json.loads(match.group(0))


class ClaudeVisionProvider(VisionProvider):
    def __init__(self) -> None:
        self._client = None

    def _get_client(self):
        if self._client is None:
            from anthropic import Anthropic

            self._client = Anthropic(
                api_key=settings.anthropic_api_key,
                base_url=settings.anthropic_base_url,
                max_retries=5, # Raise the RateLimit ceiling above the SDK default (2).
            )
        return self._client

    @property
    def is_available(self) -> bool:
        return bool(settings.anthropic_api_key)

    def _request_json(self, image: np.ndarray, system_prompt: str, user_text: str) -> dict | None:
        """Send one image + prompt to Claude and return the parsed JSON
        object, or None if encoding, the API call, or parsing failed. The
        system prompt is marked cacheable since it's identical across every
        call in a batch."""
        success, buffer = cv2.imencode(".png", image)
        if not success:
            logger.error("Failed to encode image for Claude escalation")
            return None
        image_b64 = base64.b64encode(buffer.tobytes()).decode("utf-8")

        try:
            client = self._get_client()
            message = client.messages.create(
                model=settings.anthropic_claude_model,
                max_tokens=2048,
                thinking={"type": "disabled"},
                output_config={"effort": "low"},
                system=[
                    {"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}
                ],
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
                            {"type": "text", "text": user_text},
                        ],
                    }
                ],
            )
            text_blocks = [block.text for block in message.content if block.type == "text"]
            raw_text = "".join(text_blocks).strip()
            try:
                return _extract_json_object(raw_text)
            except json.JSONDecodeError:
                logger.error(
                    "Claude vision response was not valid JSON (stop_reason=%s): %r",
                    message.stop_reason,
                    raw_text,
                )
                return None
        except Exception:
            logger.exception("Claude vision escalation failed")
            return None

    def read_submission(
        self, page_image: np.ndarray, field_names: list[str], question_numbers: list[int]
    ) -> SubmissionEscalationResult:
        empty = SubmissionEscalationResult(fields={}, answers={})
        if not self.is_available or (not field_names and not question_numbers):
            return empty

        request_parts: list[str] = []
        if field_names:
            hints = "; ".join(f"{name} ({FIELD_HINTS.get(name, name)})" for name in field_names)
            request_parts.append(f"Fields to read: {hints}.")
        if question_numbers:
            request_parts.append(f"Questions to classify: {sorted(question_numbers)}.")

        parsed = self._request_json(page_image, SUBMISSION_SYSTEM_PROMPT, " ".join(request_parts))
        if parsed is None:
            return empty

        raw_fields = parsed.get("fields")
        fields: dict[str, OCRFieldResult] = {}
        if isinstance(raw_fields, dict):
            for name in field_names:
                raw = raw_fields.get(name)
                if not isinstance(raw, dict):
                    continue
                text = raw.get("text")
                text = text.strip() if isinstance(text, str) else None
                try:
                    confidence = max(0.0, min(1.0, float(raw.get("confidence", 0.0))))
                except (TypeError, ValueError):
                    confidence = 0.0
                fields[name] = OCRFieldResult(
                    field_name=name,
                    raw_text=text or "",
                    parsed_value=text or None,
                    confidence=confidence,
                    needs_escalation=not text,
                )

        raw_answers = parsed.get("answers")
        answers: dict[int, AnswerDetection] = {}
        if isinstance(raw_answers, dict):
            for qnum in question_numbers:
                raw = raw_answers.get(str(qnum))
                if not isinstance(raw, dict):
                    continue
                answer = raw.get("answer")
                if answer not in VALID_ANSWERS:
                    answer = None
                state = raw.get("state", "ambiguous")
                if state not in {s.value for s in AnswerState}:
                    state = "ambiguous"
                try:
                    confidence = max(0.0, min(1.0, float(raw.get("confidence", 0.0))))
                except (TypeError, ValueError):
                    confidence = 0.0
                answers[qnum] = AnswerDetection(
                    answer=answer,
                    state=AnswerState(state),
                    confidence=confidence,
                    method=DetectionMethod.CLOUD_VLM,
                )

        return SubmissionEscalationResult(fields=fields, answers=answers)
