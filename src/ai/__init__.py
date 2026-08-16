"""Vision provider factory -- pipeline code should only ever import
`get_vision_provider`, never a concrete provider class."""

from __future__ import annotations

from src.ai.base import VisionProvider
from src.config import settings


def get_vision_provider() -> VisionProvider:
    provider = settings.vision_provider.lower()

    if provider == "claude":
        from src.ai.claude_provider import ClaudeVisionProvider

        return ClaudeVisionProvider()

    from src.ai.null_provider import NullVisionProvider

    return NullVisionProvider()
