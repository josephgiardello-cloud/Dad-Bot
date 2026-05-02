from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, cast

from dadbot.core.execution_contract import TurnDelivery, TurnResponse, live_turn_request


@dataclass(slots=True)
class UserMessageResult:
    reply: str
    should_end: bool
    mood: str
    pipeline: dict
    turn_health: dict = field(default_factory=dict)
    ux_feedback: dict = field(default_factory=dict)
    active_rules: list[str] = field(default_factory=list)


class RuntimeServices(Protocol):
    def handle_user_message(
        self,
        *,
        thread_id: str,
        text: str,
        attachments: list[dict] | None = None,
    ) -> UserMessageResult: ...

    def write_memory(self, *, thread_id: str, payload: dict) -> None: ...


class LLMService(Protocol):
    def generate_reply(
        self,
        *,
        thread_id: str,
        text: str,
        attachments: list[dict] | None = None,
    ) -> UserMessageResult: ...


class MemoryService(Protocol):
    def write(self, *, thread_id: str, payload: dict) -> None: ...


class DadBotLLMService:
    def __init__(self, bot) -> None:
        self.bot = bot

    def generate_reply(
        self,
        *,
        thread_id: str,
        text: str,
        attachments: list[dict] | None = None,
    ) -> UserMessageResult:
        response = self.bot.execute_turn(
            live_turn_request(
                text,
                attachments=list(attachments or []),
                delivery=TurnDelivery.SYNC,
                session_id=str(thread_id or getattr(self.bot, "active_thread_id", "") or "default"),
            ),
        )
        reply, should_end = cast(TurnResponse, response).as_result()

        mood = str(self.bot.last_saved_mood() or "neutral")
        pipeline = dict(self.bot.turn_pipeline_snapshot() or {})
        turn_health = dict(self.bot.turn_health_state() or {})
        ux_feedback = dict(self.bot.turn_ux_feedback() or {})
        active_rules = list(self.bot.profile_runtime.effective_behavior_rules())
        return UserMessageResult(
            reply=str(reply or ""),
            should_end=should_end,
            mood=mood,
            pipeline=pipeline,
            turn_health=turn_health,
            ux_feedback=ux_feedback,
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

    def __init__(
        self,
        bot,
        *,
        llm: LLMService | None = None,
        memory: MemoryService | None = None,
    ) -> None:
        self.bot = bot
        self.llm: LLMService = llm or DadBotLLMService(bot)
        self.memory: MemoryService = memory or DadBotMemoryService(bot)

    def handle_user_message(
        self,
        *,
        thread_id: str,
        text: str,
        attachments: list[dict] | None = None,
    ) -> UserMessageResult:
        return self.llm.generate_reply(
            thread_id=thread_id,
            text=text,
            attachments=attachments,
        )

    def write_memory(self, *, thread_id: str, payload: dict) -> None:
        self.memory.write(thread_id=thread_id, payload=payload)
