"""Automatic Speech Recognition (ASR) fallback for DadBot.

This implementation is deterministic and offline-safe. It treats input as
UTF-8 text when possible and returns metadata-only output for binary payloads.
"""

from __future__ import annotations

import hashlib


class ASR:
    def transcribe(self, audio_bytes: bytes) -> str:
        payload = bytes(audio_bytes or b"")
        if not payload:
            return ""

        try:
            decoded = payload.decode("utf-8").strip()
            if decoded:
                return decoded
        except UnicodeDecodeError:
            decoded = ""

        digest = hashlib.sha256(payload).hexdigest()[:12]
        return f"[audio:{len(payload)}b:{digest}]"
