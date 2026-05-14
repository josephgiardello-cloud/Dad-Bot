from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, cast

from dadbot.core.dadbot import DadBot
from dadbot.core.execution_contract import SovereignContext, TurnDelivery, TurnResponse, live_turn_request


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
        metadata: dict | None = None,
    ) -> dict:
        normalized_thread_id = str(thread_id or self._bot.active_thread_id or "default")
        if normalized_thread_id != str(self._bot.active_thread_id or ""):
            try:
                self._bot.switch_chat_thread(normalized_thread_id)
            except Exception:
                pass

        request_metadata = dict(metadata or {})
        requested_tools = list(request_metadata.get("allowed_tools") or [])
        if requested_tools:
            denied = [
                str(tool or "")
                for tool in requested_tools
                if str(tool or "") and not bool(self._bot._service_tool_allowed(str(tool or "")))
            ]
            if denied:
                blocked = ", ".join(sorted(set(denied)))
                return {
                    "reply": f"I can't use restricted tools in this mode: {blocked}.",
                    "should_end": False,
                    "mood": str(self._bot.last_saved_mood() or "neutral"),
                    "pipeline": {"mode": "tool_policy_gate", "blocked_tools": denied},
                    "turn_health": dict(self._bot.turn_health_state() or {}),
                    "ux_feedback": dict(self._bot.turn_ux_feedback() or {}),
                    "multi_agent_trace": {},
                    "photo_requested": False,
                    "tts_requested": False,
                }

        if self._should_fast_path(content, attachments, request_metadata):
            return self._run_fast_path_reply(
                thread_id=normalized_thread_id,
                content=str(content or ""),
                metadata=request_metadata,
            )

        response = self._bot.execute_turn(
            live_turn_request(
                str(content or ""),
                attachments=list(attachments or []),
                delivery=TurnDelivery.SYNC,
                context=SovereignContext(session_id=normalized_thread_id),
                metadata=request_metadata,
            ),
        )
        reply, should_end = cast(TurnResponse, response).as_result()
        turn_health = dict(self._bot.turn_health_state() or {})
        ux_feedback = dict(self._bot.turn_ux_feedback() or {})
        multi_agent_trace = self._multi_agent_trace_snapshot()
        if isinstance(metadata, dict) and metadata:
            with self._bot._session_lock:
                self._bot.ensure_chat_thread_state(preserve_active_runtime=True)
                snapshot = self._bot.normalize_thread_snapshot(self._bot.thread_snapshots.get(normalized_thread_id))
                history = [dict(item) for item in list(snapshot.get("history") or []) if isinstance(item, dict)]
                for index in range(len(history) - 1, -1, -1):
                    message = dict(history[index] or {})
                    if str(message.get("role") or "").strip().lower() != "user":
                        continue
                    existing_metadata = dict(message.get("metadata") or {})
                    for key, value in dict(metadata).items():
                        if key == "gateway":
                            gateway_payload = dict(existing_metadata.get("gateway") or {})
                            gateway_payload.update(dict(value or {}))
                            existing_metadata["gateway"] = gateway_payload
                        else:
                            existing_metadata[key] = value
                    message["metadata"] = existing_metadata
                    history[index] = message
                    break
                snapshot["history"] = history
                self._bot.thread_snapshots[normalized_thread_id] = snapshot
                if normalized_thread_id == self._bot.active_thread_id:
                    self._bot.apply_thread_snapshot_unlocked(snapshot)
                self._bot.sync_active_thread_snapshot()
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

    @staticmethod
    def _is_simple_turn(text: str) -> bool:
        normalized = str(text or "").strip().lower()
        if not normalized:
            return False
        if len(normalized) > 64:
            return False
        patterns = (
            r"^(hi|hey|hello|yo)[!. ]*$",
            r"^(thanks|thank you|thx)[!. ]*$",
            r"^(good morning|good night|good evening)[!. ]*$",
            r"^(love you|luv you)[!. ]*$",
            r"^(how are you|you there)\??$",
        )
        return any(re.match(pattern, normalized) for pattern in patterns)

    def _should_fast_path(self, text: str, attachments: list[dict] | None, metadata: dict[str, Any]) -> bool:
        if attachments:
            return False
        if bool(metadata.get("disable_fast_path", False)):
            return False
        return self._is_simple_turn(text)

    def _run_fast_path_reply(self, *, thread_id: str, content: str, metadata: dict[str, Any]) -> dict[str, Any]:
        lowered = str(content or "").strip().lower()
        if any(token in lowered for token in ("thanks", "thank you", "thx")):
            reply = "Always got your back, kiddo."
        elif any(token in lowered for token in ("love you", "luv you")):
            reply = "Love you too, buddy."
        elif "how are you" in lowered:
            reply = "Steady and ready for you, champ. What's up?"
        else:
            reply = "Hey, kiddo. I'm right here."

        with self._bot._session_lock:
            self._bot.ensure_chat_thread_state(preserve_active_runtime=True)
            snapshot = self._bot.normalize_thread_snapshot(self._bot.thread_snapshots.get(thread_id))
            history = [dict(item) for item in list(snapshot.get("history") or []) if isinstance(item, dict)]
            history.append({"role": "user", "content": str(content or ""), "metadata": dict(metadata or {})})
            history.append({"role": "assistant", "content": reply, "metadata": {"fast_path": True}})
            snapshot["history"] = history
            self._bot.thread_snapshots[thread_id] = snapshot
            if thread_id == self._bot.active_thread_id:
                self._bot.apply_thread_snapshot_unlocked(snapshot)
            self._bot.sync_active_thread_snapshot()

        return {
            "reply": reply,
            "should_end": False,
            "mood": str(self._bot.last_saved_mood() or "neutral"),
            "pipeline": {"mode": "fast_path", "reason": "simple_turn"},
            "turn_health": dict(self._bot.turn_health_state() or {}),
            "ux_feedback": dict(self._bot.turn_ux_feedback() or {}),
            "multi_agent_trace": {},
            "photo_requested": False,
            "tts_requested": False,
        }

    def list_recent_checkpoints(self, *, limit: int = 10) -> list[dict[str, Any]]:
        persistence = getattr(self._bot, "conversation_persistence", None)
        if persistence is None:
            return []
        ledger_reader = getattr(persistence, "_execution_ledger", None)
        if not callable(ledger_reader):
            return []
        try:
            events = list(ledger_reader().read())
        except Exception:
            return []
        current_session = str(getattr(self._bot, "active_thread_id", "") or "default")
        rows: list[dict[str, Any]] = []
        for event in reversed(events):
            if str(event.get("type") or "") != "GRAPH_CHECKPOINT":
                continue
            if str(event.get("session_id") or "") not in {"", current_session}:
                continue
            payload = dict(event.get("payload") or {})
            checkpoint = dict(payload.get("checkpoint") or {})
            if not checkpoint:
                continue
            rows.append(
                {
                    "trace_id": str(checkpoint.get("trace_id") or payload.get("trace_id") or ""),
                    "checkpoint_hash": str(checkpoint.get("checkpoint_hash") or ""),
                    "prev_checkpoint_hash": str(checkpoint.get("prev_checkpoint_hash") or ""),
                    "phase": str(checkpoint.get("phase") or ""),
                    "event_sequence_id": int(checkpoint.get("event_sequence_id") or 0),
                }
            )
            if len(rows) >= max(1, int(limit or 10)):
                break
        return rows

    def restore_checkpoint(self, *, trace_id: str) -> bool:
        persistence = getattr(self._bot, "conversation_persistence", None)
        resume = getattr(persistence, "resume_graph_checkpoint", None)
        if not callable(resume):
            return False
        try:
            restored = resume(trace_token=str(trace_id or "").strip())
        except Exception:
            return False
        return isinstance(restored, dict) and bool(restored)

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
        self._configure_ui_runtime_mode()
        self.api = UIRuntimeAPI(bot)

    def _configure_ui_runtime_mode(self) -> None:
        """Keep the Streamlit surface resilient when graph execution encounters transient failures."""
        config = getattr(self.bot, "config", None)
        if config is not None and hasattr(config, "strict_graph_mode"):
            config.strict_graph_mode = False
        self.bot._strict_graph_mode = False
        orchestrator = getattr(self.bot, "_turn_orchestrator", None)
        if orchestrator is not None and hasattr(orchestrator, "_strict"):
            orchestrator._strict = False

    @classmethod
    def build(cls) -> StreamlitRuntime:
        return cls(DadBot())

    def send_user_message(
        self,
        *,
        thread_id: str,
        content: str,
        attachments: list[dict] | None = None,
        metadata: dict | None = None,
    ) -> dict:
        return self.api.send_user_message(
            thread_id=thread_id,
            content=content,
            attachments=attachments,
            metadata=metadata,
        )
