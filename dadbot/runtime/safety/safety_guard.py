from __future__ import annotations

from typing import Callable


class SafetyGuard:
    def _handle_pii(self, text: str, *, scrubber: Callable[[str], tuple[str, list[str]]]) -> tuple[str, list[str]]:
        return scrubber(str(text or ""))

    @staticmethod
    def _sanitize_input(text: str, *, max_len: int = 16000) -> tuple[str, bool, bool]:
        raw = str(text or "")
        cleaned_chars: list[str] = []
        removed_control = False
        for ch in raw:
            if ch in {"\n", "\r", "\t"}:
                cleaned_chars.append(ch)
                continue
            if ord(ch) < 32:
                removed_control = True
                continue
            cleaned_chars.append(ch)
        normalized = "".join(cleaned_chars).strip()
        truncated = False
        if len(normalized) > int(max_len):
            normalized = normalized[: int(max_len)]
            truncated = True
        return normalized, truncated, removed_control

    def _scrub_output(self, text: str, *, scrubber: Callable[[str], tuple[str, list[str]]]) -> tuple[str, list[str]]:
        scrubbed, pii_types = self._handle_pii(str(text or ""), scrubber=scrubber)
        return scrubbed, list(pii_types)

    def sanitize_text(self, text: str, *, scrubber: Callable[[str], tuple[str, list[str]]], max_len: int = 16000) -> tuple[str, dict]:
        normalized, truncated, removed_control = self._sanitize_input(text, max_len=max_len)
        scrubbed, pii_types = self._scrub_output(normalized, scrubber=scrubber)
        return scrubbed, {
            "pii_types": list(pii_types),
            "truncated": truncated,
            "control_chars_removed": removed_control,
        }
