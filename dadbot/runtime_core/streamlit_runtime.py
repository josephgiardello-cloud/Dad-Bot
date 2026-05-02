from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

from dadbot.core.dadbot import DadBot
from dadbot.core.execution_contract import TurnDelivery, TurnResponse, live_turn_request


@dataclass
class ThreadView:
    thread_id: str
    messages: list[dict] = field(default_factory=list)
    thinking: dict = field(default_factory=dict)
    decision: dict = field(default_factory=dict)
    turn_graph: dict = field(default_factory=dict)
    execution_boundaries: list[dict] = field(default_factory=list)
    view_version: str = "v2"
    schema_policy: dict = field(default_factory=dict)


class UIRuntimeAPI:
    """Compatibility API exposed to Streamlit surfaces.

    This adapter preserves the long-lived UI method surface while delegating
    behavior to the current DadBot runtime facade.
    """

    def __init__(self, bot: DadBot) -> None:
        self._bot = bot

    def __getattr__(self, name: str) -> Any:
        if name == "get_view":
            return self._compat_get_view
        return getattr(self._bot, name)

    def seed_thread(self, thread_id: str, messages: list[dict] | None) -> None:
        normalized_thread_id = str(thread_id or "default").strip() or "default"
        with self._bot._session_lock:
            self._bot.ensure_chat_thread_state(preserve_active_runtime=True)
            snapshot = self._bot.normalize_thread_snapshot(
                self._bot.thread_snapshots.get(normalized_thread_id),
            )
            snapshot["history"] = [dict(item) for item in list(messages or []) if isinstance(item, dict)]
            self._bot.thread_snapshots[normalized_thread_id] = snapshot
            if normalized_thread_id == self._bot.active_thread_id:
                self._bot.apply_thread_snapshot_unlocked(snapshot)
            self._bot.sync_active_thread_snapshot()

    def snapshot_thread_messages(
        self,
        thread_id: str,
        *,
        default_greeting: str = "",
    ) -> list[dict]:
        normalized_thread_id = str(thread_id or "default").strip() or "default"
        with self._bot._session_lock:
            self._bot.ensure_chat_thread_state(preserve_active_runtime=True)
            snapshot = self._bot.normalize_thread_snapshot(
                self._bot.thread_snapshots.get(normalized_thread_id),
            )
            history = [dict(item) for item in list(snapshot.get("history") or []) if isinstance(item, dict)]
            if not history:
                history = [
                    {
                        "role": "assistant",
                        "content": self._bot.opening_message(
                            default_greeting or "That's my boy. I love hearing that, Tony.",
                        ),
                    },
                ]
                snapshot["history"] = history
                self._bot.thread_snapshots[normalized_thread_id] = snapshot
                if normalized_thread_id == self._bot.active_thread_id:
                    self._bot.apply_thread_snapshot_unlocked(snapshot)
                self._bot.sync_active_thread_snapshot()
        return history

    def _compat_get_view(self, thread_id: str, *, version: str = "v2") -> ThreadView:
        normalized_thread_id = str(thread_id or "default").strip() or "default"
        with self._bot._session_lock:
            self._bot.ensure_chat_thread_state(preserve_active_runtime=True)
            snapshot = self._bot.normalize_thread_snapshot(
                self._bot.thread_snapshots.get(normalized_thread_id),
            )
        planner_debug = dict(snapshot.get("planner_debug") or {})
        messages = [dict(item) for item in list(snapshot.get("history") or []) if isinstance(item, dict)]
        thinking = {
            "final_path": str(planner_debug.get("final_path") or "model_reply"),
            "reply_source": str(planner_debug.get("reply_source") or ""),
            "mood_detected": str(self._bot.last_saved_mood() or "neutral"),
            "active_rules": list(planner_debug.get("active_rules") or []),
            "pipeline_steps": list(planner_debug.get("pipeline_steps") or []),
            "turn_health": dict(self._bot.turn_health_state() or {}),
            "ux_feedback": dict(self._bot.turn_ux_feedback() or {}),
        }
        return ThreadView(
            thread_id=normalized_thread_id,
            messages=messages,
            thinking=thinking,
            decision={},
            turn_graph={},
            execution_boundaries=[],
            view_version=str(version or "v2"),
            schema_policy={},
        )

    def _multi_agent_trace_snapshot(self) -> dict[str, Any]:
        """Extract delegation/reasoning details from the latest turn context for UI visibility."""
        orchestrator = getattr(self._bot, "turn_orchestrator", None)
        context = getattr(orchestrator, "_last_turn_context", None)
        if context is None:
            return {}

        metadata = dict(getattr(context, "metadata", {}) or {})
        state = dict(getattr(context, "state", {}) or {})

        arbitration = dict(state.get("arbitration_metadata") or {})
        blackboard = dict(state.get("agent_blackboard") or {})
        delegation_results = list(state.get("delegation_results") or [])
        reasoning_steps = list(state.get("reasoning_steps") or [])

        if not arbitration and not blackboard and not delegation_results and not reasoning_steps:
            return {}

        return {
            "delegation_depth": int(metadata.get("delegation_depth") or 0),
            "subtasks_executed": int(metadata.get("subtasks_executed") or 0),
            "arbitration": arbitration,
            "blackboard": blackboard,
            "delegation_results": delegation_results,
            "reasoning_steps": reasoning_steps,
        }

    def send_user_message(
        self,
        *,
        thread_id: str,
        content: str,
        attachments: list[dict] | None = None,
    ) -> dict:
        normalized_thread_id = str(thread_id or self._bot.active_thread_id or "default")
        if normalized_thread_id != str(self._bot.active_thread_id or ""):
            try:
                self._bot.switch_chat_thread(normalized_thread_id)
            except Exception:
                pass
        response = self._bot.execute_turn(
            live_turn_request(
                str(content or ""),
                attachments=list(attachments or []),
                delivery=TurnDelivery.SYNC,
                session_id=normalized_thread_id,
            ),
        )
        reply, should_end = cast(TurnResponse, response).as_result()
        turn_health = dict(self._bot.turn_health_state() or {})
        ux_feedback = dict(self._bot.turn_ux_feedback() or {})
        multi_agent_trace = self._multi_agent_trace_snapshot()
        return {
            "reply": str(reply or ""),
            "should_end": bool(should_end),
            "mood": str(self._bot.last_saved_mood() or "neutral"),
            "pipeline": dict(self._bot.turn_pipeline_snapshot() or {}),
            "turn_health": turn_health,
            "ux_feedback": ux_feedback,
            "multi_agent_trace": multi_agent_trace,
            "photo_requested": False,
            "tts_requested": False,
        }

    def process_until_idle(self, *, max_events: int = 256) -> list[dict]:
        _ = max_events
        return []

    def emit_assistant_attachment(self, *, thread_id: str, attachment: dict) -> None:
        normalized_thread_id = str(thread_id or self._bot.active_thread_id or "default")
        with self._bot._session_lock:
            self._bot.ensure_chat_thread_state(preserve_active_runtime=True)
            snapshot = self._bot.normalize_thread_snapshot(
                self._bot.thread_snapshots.get(normalized_thread_id),
            )
            history = [dict(item) for item in list(snapshot.get("history") or []) if isinstance(item, dict)]
            assistant_indexes = [index for index, message in enumerate(history) if message.get("role") == "assistant"]
            if assistant_indexes:
                target = history[assistant_indexes[-1]]
                attachments = list(target.get("attachments") or [])
                attachments.append(dict(attachment or {}))
                target["attachments"] = attachments
                history[assistant_indexes[-1]] = target
            snapshot["history"] = history
            self._bot.thread_snapshots[normalized_thread_id] = snapshot
            if normalized_thread_id == self._bot.active_thread_id:
                self._bot.apply_thread_snapshot_unlocked(snapshot)
            self._bot.sync_active_thread_snapshot()

    def emit_assistant_photo_message(
        self,
        *,
        thread_id: str,
        text: str,
        attachment: dict,
    ) -> None:
        normalized_thread_id = str(thread_id or self._bot.active_thread_id or "default")
        with self._bot._session_lock:
            self._bot.ensure_chat_thread_state(preserve_active_runtime=True)
            snapshot = self._bot.normalize_thread_snapshot(
                self._bot.thread_snapshots.get(normalized_thread_id),
            )
            history = [dict(item) for item in list(snapshot.get("history") or []) if isinstance(item, dict)]
            history.append(
                {
                    "role": "assistant",
                    "content": str(text or ""),
                    "attachments": [dict(attachment or {})],
                },
            )
            snapshot["history"] = history
            self._bot.thread_snapshots[normalized_thread_id] = snapshot
            if normalized_thread_id == self._bot.active_thread_id:
                self._bot.apply_thread_snapshot_unlocked(snapshot)
            self._bot.sync_active_thread_snapshot()

    def onboarding_complete(self) -> bool:
        profile = dict(getattr(self._bot, "PROFILE", {}) or {})
        ui = dict(profile.get("ui") or {})
        return bool(ui.get("onboarding_complete", True))

    def set_onboarding_complete(self, value: bool) -> None:
        profile = dict(getattr(self._bot, "PROFILE", {}) or {})
        ui = dict(profile.get("ui") or {})
        ui["onboarding_complete"] = bool(value)
        profile["ui"] = ui
        self._bot.PROFILE = profile


class StreamlitRuntime:
    def __init__(self, bot: DadBot) -> None:
        self.bot = bot
        self.api = UIRuntimeAPI(bot)

    @classmethod
    def build(cls) -> StreamlitRuntime:
        return cls(DadBot())

    def send_user_message(
        self,
        *,
        thread_id: str,
        content: str,
        attachments: list[dict] | None = None,
    ) -> dict:
        return self.api.send_user_message(
            thread_id=thread_id,
            content=content,
            attachments=attachments,
        )
