"""Text-to-Speech (TTS) fallback for DadBot.

This implementation is deterministic and offline-safe. It emits a compact
UTF-8 payload suitable for tests and local integrations without audio deps.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


class TTS:
    def synthesize(self, text: str) -> bytes:
        normalized = str(text or "").strip()
        if not normalized:
            return b""
        return f"TTS:{normalized}".encode("utf-8")


def resolve_piper_executable() -> str:
    executable = shutil.which("piper")
    if executable:
        return executable
    bundled_name = "piper.exe" if os.name == "nt" else "piper"
    bundled = Path(sys.executable).parent / bundled_name
    return str(bundled) if bundled.is_file() else ""
