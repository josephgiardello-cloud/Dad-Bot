from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from .policy import PolicyEngine


@dataclass(slots=True)
class UserMessageResult:
    reply: str
    should_end: bool
    mood: str
    pipeline: dict
    active_rules: list[str] = field(default_factory=list)


class RuntimeServices(Protocol):
    def handle_user_message(self, *, thread_id: str, text: str, attachments: list[dict] | None = None) -> UserMessageResult:
        ...

    def write_memory(self, *, thread_id: str, payload: dict) -> None:
        ...


class LLMService(Protocol):
    def generate_reply(self, *, thread_id: str, text: str, attachments: list[dict] | None = None) -> UserMessageResult:
        ...


class MemoryService(Protocol):
    def write(self, *, thread_id: str, payload: dict) -> None:
        ...





class DadBotLLMService:
    def __init__(self, bot) -> None:
        self.bot = bot

    def generate_reply(self, *, thread_id: str, text: str, attachments: list[dict] | None = None) -> UserMessageResult:
        reply_value = self.bot.process_user_message(text, attachments=attachments)
        if hasattr(reply_value, "reply"):
            reply = str(reply_value.reply or "")
            should_end = bool(getattr(reply_value, "should_end", False))
        elif isinstance(reply_value, tuple):
            reply = str(reply_value[0] or "")
            should_end = bool(reply_value[1]) if len(reply_value) > 1 else False
        else:
            reply = str(reply_value or "")
            should_end = False

        mood = str(self.bot.last_saved_mood() or "neutral")
        pipeline = dict(self.bot.turn_pipeline_snapshot() or {})
        active_rules = list(self.bot.profile_runtime.effective_behavior_rules())
        return UserMessageResult(
            reply=reply,
            should_end=should_end,
            mood=mood,
            pipeline=pipeline,
            active_rules=active_rules,
        )


class DadBotMemoryService:
    def __init__(self, bot) -> None:
        self.bot = bot

    def write(self, *, thread_id: str, payload: dict) -> None:
        summary = str(payload.get("summary") or "").strip()
        if not summary:
            return
        try:
            self.bot.add_memory(summary)
        except Exception:
            return


class DadBotRuntimeServices:
    """Composed service facade for the runtime core."""

    def __init__(self, bot, *, llm: LLMService | None = None, memory: MemoryService | None = None) -> None:
        self.bot = bot
        self.llm: LLMService = llm or DadBotLLMService(bot)
        self.memory: MemoryService = memory or DadBotMemoryService(bot)

    def handle_user_message(self, *, thread_id: str, text: str, attachments: list[dict] | None = None) -> UserMessageResult:
        return self.llm.generate_reply(thread_id=thread_id, text=text, attachments=attachments)

    def write_memory(self, *, thread_id: str, payload: dict) -> None:
        self.memory.write(thread_id=thread_id, payload=payload)
