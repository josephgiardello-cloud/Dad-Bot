from __future__ import annotations

from dadbot.core.facade_utils import DadBotFacadeUtils


class DadBotConvenienceMixin:
    """Small mixin for low-level convenience wrappers kept for compatibility."""

    def deliver_status_message(self, message, status_callback=None):
        DadBotFacadeUtils.deliver_status_message(self, message, status_callback=status_callback)

    @staticmethod
    def terminal_width():
        return DadBotFacadeUtils.terminal_width()

    def print_system_message(self, message):
        DadBotFacadeUtils.print_system_message(message)

    @staticmethod
    def print_speaker_message(speaker, message):
        DadBotFacadeUtils.print_speaker_message(speaker, message)

    def build_system_prompt(self) -> str:
        return DadBotFacadeUtils.build_system_prompt(self)

    def flatten_memory_payload(self, payload):
        return DadBotFacadeUtils.flatten_memory_payload(payload)

    def coerce_memory_summary(self, value):
        return DadBotFacadeUtils.coerce_memory_summary(value)

    def new_chat_session(self):
        return DadBotFacadeUtils.new_chat_session(self)
