from __future__ import annotations

import shutil
import textwrap
from typing import Any


class DadBotFacadeUtils:
    """Small helper surface for low-level facade utility routines."""

    @staticmethod
    def flatten_memory_payload(payload) -> list[Any]:
        if isinstance(payload, list):
            flattened: list[Any] = []
            for item in payload:
                flattened.extend(DadBotFacadeUtils.flatten_memory_payload(item))
            return flattened
        return [payload]

    @staticmethod
    def coerce_memory_summary(value) -> str:
        if isinstance(value, str):
            return value.strip()

        if isinstance(value, list):
            parts = [DadBotFacadeUtils.coerce_memory_summary(item) for item in value]
            parts = [part for part in parts if part]
            return "; ".join(parts)

        if isinstance(value, dict):
            if "summary" in value:
                return DadBotFacadeUtils.coerce_memory_summary(value.get("summary"))
            parts = [DadBotFacadeUtils.coerce_memory_summary(item) for item in value.values()]
            parts = [part for part in parts if part]
            return "; ".join(parts)

        if value is None:
            return ""

        return str(value).strip()

    @staticmethod
    def build_system_prompt(bot) -> str:
        return "\n\n".join(
            section
            for section in (
                bot.context_builder.build_core_persona_prompt(),
                bot.context_builder.build_dynamic_profile_context(),
            )
            if section
        )

    @staticmethod
    def new_chat_session(bot) -> list[dict[str, str]]:
        return [{"role": "system", "content": bot.build_core_persona_prompt()}]

    @staticmethod
    def terminal_width() -> int:
        return max(60, min(shutil.get_terminal_size((88, 20)).columns - 1, 100))

    @staticmethod
    def print_system_message(message: str) -> None:
        print(textwrap.fill(message, width=DadBotFacadeUtils.terminal_width()))

    @staticmethod
    def print_speaker_message(speaker: str, message: str) -> None:
        prefix = f"{speaker}: "
        wrapped = textwrap.fill(
            f"{prefix}{message}",
            width=DadBotFacadeUtils.terminal_width(),
            subsequent_indent=" " * len(prefix),
        )
        print(wrapped)
        print()

    @staticmethod
    def deliver_status_message(bot, message: str, status_callback=None) -> None:
        if status_callback is None:
            DadBotFacadeUtils.print_speaker_message("Dad", message)
            return
        status_callback(message)
