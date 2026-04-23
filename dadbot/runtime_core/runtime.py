from __future__ import annotations

from .bus import EventBus
from .models import Event, new_event
from .policy import PolicyEngine
from .services import RuntimeServices
from .store import ConversationStore


class AgentRuntime:
    """Deterministic event loop. No Streamlit imports."""

    def __init__(self, services: RuntimeServices, store: ConversationStore, *, policy_engine: PolicyEngine | None = None) -> None:
        self.services = services
        self.store = store
        self.policy_engine = policy_engine or PolicyEngine()

    def handle_event(self, event: Event) -> list[Event]:
        if event.type == "user_message":
            self.store.apply_event(event)
            return self._handle_user_message(event)

        if event.type in {"assistant_reply", "assistant_attachment_added", "thinking_update"}:
            self.store.apply_event(event)
            return []

        if event.type == "memory_write":
            self.services.write_memory(thread_id=event.thread_id, payload=dict(event.payload or {}))
            return []

        if event.type in {"tool_result", "tool_call", "thread_switch", "photo_request", "tts_request", "mood_update"}:
            return []

        return []

    def _handle_user_message(self, event: Event) -> list[Event]:
        result = self.services.handle_user_message(
            thread_id=event.thread_id,
            text=str(event.payload.get("text") or ""),
            attachments=list(event.payload.get("attachments") or []),
        )
        
        # Evaluate policies for this turn
        policies = self.policy_engine.evaluate(
            mood=str(result.mood or "neutral"),
            thread_id=event.thread_id,
            reply_text=str(result.reply or ""),
        )
        
        events: list[Event] = [
            new_event(
                "assistant_reply",
                thread_id=event.thread_id,
                payload={
                    "text": result.reply,
                    "should_end": bool(result.should_end),
                    "mood": str(result.mood or "neutral"),
                    "pipeline": dict(result.pipeline or {}),
                    "attachments": [],
                },
            )
        ,
            new_event(
                "thinking_update",
                thread_id=event.thread_id,
                payload={
                    "mood_detected": str(result.pipeline.get("current_mood") or result.mood or "neutral"),
                    "final_path": str(result.pipeline.get("final_path") or "model_reply"),
                    "reply_source": str(result.pipeline.get("reply_source") or "model_generation"),
                    "pipeline_steps": list(result.pipeline.get("steps") or []),
                    "active_rules": list(result.active_rules or []),
                },
            )
        ]
        
        # Photo request based on policy
        if policies.should_generate_photo:
            events.append(
                new_event(
                    "photo_request",
                    thread_id=event.thread_id,
                    payload={"reason": "mood_support", "mood": result.mood},
                )
            )
        
        # TTS request based on policy
        if policies.should_request_tts:
            events.append(
                new_event(
                    "tts_request",
                    thread_id=event.thread_id,
                    payload={"text": str(result.reply or "")},
                )
            )
        
        return events

    def run_until_idle(self, bus: EventBus, *, max_events: int = 256) -> list[Event]:
        processed: list[Event] = []
        budget = max(1, int(max_events or 1))
        while budget > 0 and not bus.empty():
            budget -= 1
            event = bus.next()
            processed.append(event)
            follow_up = self.handle_event(event)
            for created in follow_up:
                bus.emit(created)
        return processed
