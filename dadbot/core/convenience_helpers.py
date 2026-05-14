from __future__ import annotations

from dadbot.core.facade_utils import DadBotFacadeUtils


class ConvenienceHelpers:
    """Small composed helper for low-level convenience wrappers."""

    def __init__(self, bot) -> None:
        self._bot = bot

    def deliver_status_message(self, message, status_callback=None):
        DadBotFacadeUtils.deliver_status_message(
            self._bot,
            message,
            status_callback=status_callback,
        )

    @staticmethod
    def terminal_width():
        return DadBotFacadeUtils.terminal_width()

    def print_system_message(self, message):
        DadBotFacadeUtils.print_system_message(message)

    @staticmethod
    def print_speaker_message(speaker, message):
        DadBotFacadeUtils.print_speaker_message(speaker, message)

    def build_system_prompt(self) -> str:
        return DadBotFacadeUtils.build_system_prompt(self._bot)

    @staticmethod
    def flatten_memory_payload(payload):
        return DadBotFacadeUtils.flatten_memory_payload(payload)

    @staticmethod
    def coerce_memory_summary(value):
        return DadBotFacadeUtils.coerce_memory_summary(value)

    def new_chat_session(self):
        return DadBotFacadeUtils.new_chat_session(self._bot)
