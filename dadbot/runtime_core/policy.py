"""Policy engine for runtime decisions (photo generation, TTS, etc.)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(slots=True)
class PolicyDecisions:
    """Decisions made by the policy engine for a turn."""

    should_generate_photo: bool
    should_request_tts: bool


class PhotoPolicy(Protocol):
    """Protocol for photo generation policy."""

    def should_generate(self, *, mood: str, thread_id: str) -> bool:
        ...


class TTSPolicy(Protocol):
    """Protocol for TTS request policy."""

    def should_request(self, *, thread_id: str, reply_text: str) -> bool:
        ...


class DefaultPhotoPolicy:
    """Photo generation policy: support mood states."""

    def should_generate(self, *, mood: str, thread_id: str) -> bool:
        """Generate photos for supportive moods (sad, stressed, tired)."""
        return str(mood or "").strip().lower() in {"sad", "stressed", "tired"}


class DefaultTTSPolicy:
    """TTS request policy: non-empty replies."""

    def should_request(self, *, thread_id: str, reply_text: str) -> bool:
        """Request TTS for non-empty replies."""
        return bool(str(reply_text or "").strip())


class PolicyEngine:
    """Decision engine for runtime policies."""

    def __init__(
        self,
        *,
        photo_policy: PhotoPolicy | None = None,
        tts_policy: TTSPolicy | None = None,
    ) -> None:
        self.photo_policy = photo_policy or DefaultPhotoPolicy()
        self.tts_policy = tts_policy or DefaultTTSPolicy()

    def evaluate(self, *, mood: str, thread_id: str, reply_text: str) -> PolicyDecisions:
        """Evaluate all policies for a turn and return decisions."""
        return PolicyDecisions(
            should_generate_photo=self.photo_policy.should_generate(mood=mood, thread_id=thread_id),
            should_request_tts=self.tts_policy.should_request(thread_id=thread_id, reply_text=reply_text),
        )
