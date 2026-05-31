"""Text-to-Speech (TTS) fallback for DadBot.

This implementation is deterministic and offline-safe. It emits a compact
UTF-8 payload suitable for tests and local integrations without audio deps.
"""

from __future__ import annotations


class TTS:
    def synthesize(self, text: str) -> bytes:
        normalized = str(text or "").strip()
        if not normalized:
            return b""
        return f"TTS:{normalized}".encode("utf-8")
