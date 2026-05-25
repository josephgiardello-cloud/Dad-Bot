"""dadbot/core/dadbot.py — DadBot public facade.

DadBot is intentionally thin.  Logic lives in dedicated managers held by
``self.services`` and in the five behaviour mixins:

    DadBotBootMixin     boot/init/shutdown lifecycle
    DadBotTurnMixin     turn execution and graph failure handling
    DadBotLlmMixin      LLM/model call forwarding
    DadBotMcpMixin      local MCP server management
    DadBotHealthMixin   health/UX state and checkpoint/replay

The remaining body of this class is pure delegation plumbing:
    * Class-level routing maps (_CONFIG_ATTR_MAP, explicit service names, etc.)
    * __getattr__ / __setattr__ for zero-overhead manager attribute routing
    * Explicit @property getters/setters for every registered manager
    * Config and runtime-state property aliases (PROFILE, MEMORY_STORE, ...)
    * model_port / ux_gateway properties (access private attrs set by boot mixin)
    * reset_session_state and two small turn-state helpers
"""

from __future__ import annotations

import logging
import os
import time
import warnings
from typing import Any, cast
from pathlib import Path
from dadbot.typing import MemoryManager, RelationshipManager, MoodManager, ProfileRuntime, EventBus

try:
    import ollama
except ImportError:  # pragma: no cover
    ollama = None  # type: ignore[assignment]

from dadbot.assistant_runtime import AssistantRuntime
from dadbot.core.action_mixin import DadBotActionMixin
from dadbot.core.boot_mixin import DadBotBootMixin
from dadbot.core.compat_mixin import DadBotCompatMixin
from dadbot.core.health_mixin import DadBotHealthMixin
from dadbot.core.llm_mixin import DadBotLlmMixin
from dadbot.core.services import build_services
from dadbot.core.mcp_mixin import DadBotMcpMixin
from dadbot.core.turn_mixin import DadBotTurnMixin
from dadbot.core.execution_contract import ExecutionEntry
from dadbot.core.runtime_errors import ReplayInvariantViolation
from dadbot.core.ux_projection_gateway import TurnUxProjectionGateway
from dadbot.runtime.model import ModelPort

if ollama is None:
    logging.getLogger(__name__).warning(
        "ollama package is not installed; Ollama-backed features will be unavailable. Install with: pip install ollama",
    )

logger = logging.getLogger(__name__)






class DadBot(
    DadBotBootMixin,
    DadBotTurnMixin,
    DadBotLlmMixin,
    DadBotMcpMixin,
    DadBotHealthMixin,
    DadBotCompatMixin,
    DadBotActionMixin,
):
    def __init__(
        self,
        *,
        memory_manager: Any,
        relationship_manager: Any,
        mood_manager: Any,
        profile_runtime: Any,
        event_bus: Any,
        **kwargs: Any,
    ):
        # Set manager attributes BEFORE calling boot mixin
        if 'services' in kwargs and kwargs['services'] is not None:
            self.services = kwargs.pop('services')
        else:
            self.services = build_services()
        self._memory_manager = memory_manager
        self._relationship_manager = relationship_manager
        self._mood_manager = mood_manager
        self._profile_runtime = profile_runtime
        self._event_bus = event_bus
        # Now safe to call super (which may access self.memory etc.)
        super().__init__(**kwargs)
    # ------------------------------------------------------------------
    # Explicit properties for config attributes (formerly unified routing)
    # ------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        return self.config.model_name
    
    @property
    def tool_registry(self):
        return self.services.tool_registry

    @property
    def metrics(self):
        return self.services.metrics

    @property
    def planner(self):
        return self.services.planner

    @property
    def smart_home(self):
        return getattr(self.services, 'smart_home', None)

    @property
    def asr(self):
        return getattr(self.services, 'asr', None)

    @property
    def tts(self):
        return getattr(self.services, 'tts', None)

    @property
    def fallback_models(self):
        return self.config.fallback_models


    @property
    def active_model(self) -> str:
        return self.config.active_model

    @property
    def active_embedding_model(self) -> str:
        return self.config.active_embedding_model or ""

    @property
    def llm_provider(self) -> str:
        return self.config.llm_provider

    @property
    def llm_model(self) -> str:
        return self.config.llm_model

    @property
    def tenant_id(self) -> str:
        return self.config.tenant_id

    @property
    def append_signoff(self) -> bool:
        return self.config.append_signoff

    @property
    def light_mode(self) -> bool:
        return self.config.light_mode

    @property
    def preferred_embedding_models(self) -> list[str]:
        return list(self.config.preferred_embedding_models)

    @property
    def recent_history_window(self) -> int:
        return self.config.recent_history_window

    @property
    def max_history_messages_scan(self) -> int:
        return self.config.max_history_messages_scan

    @property
    def summary_trigger_messages(self) -> int:
        return self.config.summary_trigger_messages

    @property
    def relationship_reflection_interval(self) -> int:
        return self.config.relationship_reflection_interval

    @property
    def context_token_budget(self) -> int:
        return self.config.context_token_budget

    @property
    def reserved_response_tokens(self) -> int:
        return self.config.reserved_response_tokens

    @property
    def approx_chars_per_token(self) -> float:
        return self.config.approx_chars_per_token

    @property
    def mood_detection_temperature(self) -> float:
        return self.config.mood_detection_temperature


    @property
    def stream_timeout_seconds(self) -> float:
        return self.config.stream_timeout_seconds

    @property
    def stream_max_chars(self) -> int:
        return self.config.stream_max_chars

    @property
    def profile_path(self) -> str:
        return str(self.config.profile_path)

    @property
    def memory_path(self) -> str:
        return str(self.config.memory_path)

    @property
    def semantic_memory_db_path(self) -> str:
        return str(self.config.semantic_memory_db_path)

    @property
    def graph_store_db_path(self) -> str:
        return str(self.config.graph_store_db_path)

    @property
    def session_log_dir(self) -> str:
        return str(self.config.session_log_dir)

    # ------------------------------------------------------------------
    # Explicit properties for runtime state attributes
    # ------------------------------------------------------------------


    @property
    def runtime_state_container(self) -> Any:
        return self._memory_manager.container if hasattr(self._memory_manager, 'container') else None

    @property
    def history(self) -> Any:
        return self._memory_manager.history if hasattr(self._memory_manager, 'history') else None

    @property
    def session_moods(self) -> Any:
        return self._mood_manager.session_moods if hasattr(self._mood_manager, 'session_moods') else None

    @property
    def session_summary(self) -> Any:
        return self._memory_manager.session_summary if hasattr(self._memory_manager, 'session_summary') else None

    @property
    def session_summary_updated_at(self) -> Any:
        return self._memory_manager.session_summary_updated_at if hasattr(self._memory_manager, 'session_summary_updated_at') else None

    @property
    def session_summary_covered_messages(self) -> Any:
        return self._memory_manager.session_summary_covered_messages if hasattr(self._memory_manager, 'session_summary_covered_messages') else None

    @property
    def last_relationship_reflection_turn(self) -> Any:
        return self._relationship_manager.last_relationship_reflection_turn if hasattr(self._relationship_manager, 'last_relationship_reflection_turn') else None


    @property
    def pending_daily_checkin_context(self) -> Any:
        return self._mood_manager.pending_daily_checkin_context if hasattr(self._mood_manager, 'pending_daily_checkin_context') else None

    @property
    def active_tool_observation_context(self) -> Any:
        return self._memory_manager.active_tool_observation_context if hasattr(self._memory_manager, 'active_tool_observation_context') else None

    @property
    def planner_debug(self) -> Any:
        return self._memory_manager.planner_debug if hasattr(self._memory_manager, 'planner_debug') else None

    @property
    def chat_threads(self) -> Any:
        return self._memory_manager.chat_threads if hasattr(self._memory_manager, 'chat_threads') else None

    @property
    def active_thread_id(self) -> Any:
        return self._memory_manager.active_thread_id if hasattr(self._memory_manager, 'active_thread_id') else None

    @property
    def thread_snapshots(self) -> Any:
        return self._memory_manager.thread_snapshots if hasattr(self._memory_manager, 'thread_snapshots') else None

    # ------------------------------------------------------------------
    # Explicit properties for internal runtime attributes
    # ------------------------------------------------------------------


    @property
    def prompt_guard_stats(self) -> Any:
        return self._internal_runtime.prompt_guard_stats

    @property
    def last_memory_context_stats(self) -> Any:
        return self._internal_runtime.last_memory_context_stats

    @property
    def last_output_moderation(self) -> Any:
        return self._internal_runtime.last_output_moderation

    @property
    def last_reply_supervisor(self) -> Any:
        return self._internal_runtime.last_reply_supervisor

    @property
    def last_turn_pipeline(self) -> Any:
        return self._internal_runtime.last_turn_pipeline

    @property
    def last_turn_health_state(self) -> Any:
        return self._internal_runtime.last_turn_health_state

    @property
    def last_turn_ux_feedback(self) -> Any:
        return self._internal_runtime.last_turn_ux_feedback

    @property
    def shadow_decision_bus(self) -> Any:
        return self._internal_runtime.shadow_decision_bus

    @property
    def last_shadow_decision_report(self) -> Any:
        return self._internal_runtime.last_shadow_decision_report

    @property
    def background_task_ids(self) -> Any:
        return self._internal_runtime.background_task_ids

    # (Add more explicit properties as needed for full coverage)
    # ------------------------------------------------------------------
    # Agentic Planning/Reasoning Loop (ReAct-style)
    # ------------------------------------------------------------------

    """Thin public facade for the Dad Bot persona.

    This facade is intentionally thin -- prefer direct manager access in new code.

    Architecture overview
    ---------------------
    DadBot is the public facade for all bot functionality.  Heavy logic lives in
    dedicated managers held by ``self.services`` and in the five behaviour mixins
    listed at the top of this module.  The facade keeps the high-level public API
    stable while delegating most implementation detail to the container.

    Delegation strategies
    ---------------------
    1. **Behaviour mixins** -- grouped logical concerns (boot, turn, LLM, MCP, health).
    2. **Explicit @property delegation** -- thin shims for every registered manager so
       that IDE discoverability and static analysis work.
     3. **Indexed service/provider lookup** -- unknown attribute lookups resolve through
         the service container by explicit service name or provider ownership.
    """


    # --- All dynamic attribute routing, unified routing, and deprecated alias logic removed. ---





    def deliver_status_message(self, message: str, status_callback: Any = None) -> None:
        from dadbot.core.facade_utils import DadBotFacadeUtils
        DadBotFacadeUtils.deliver_status_message(self, message, status_callback=status_callback)



    @staticmethod
    def terminal_width() -> int:
        from dadbot.core.facade_utils import DadBotFacadeUtils
        return DadBotFacadeUtils.terminal_width()



    def print_system_message(self, message: str) -> None:
        from dadbot.core.facade_utils import DadBotFacadeUtils
        DadBotFacadeUtils.print_system_message(message)



    @staticmethod
    def print_speaker_message(speaker: str, message: str) -> None:
        from dadbot.core.facade_utils import DadBotFacadeUtils
        DadBotFacadeUtils.print_speaker_message(speaker, message)



    def build_system_prompt(self) -> str:
        from dadbot.core.facade_utils import DadBotFacadeUtils
        return DadBotFacadeUtils.build_system_prompt(self)

    def build_cross_session_context(self, user_input: str = "") -> str | None:
        return self.context_builder.build_cross_session_context(user_input)



    @staticmethod
    def flatten_memory_payload(payload: Any) -> Any:
        from dadbot.core.facade_utils import DadBotFacadeUtils
        return DadBotFacadeUtils.flatten_memory_payload(payload)



    @staticmethod
    def coerce_memory_summary(value: Any) -> Any:
        from dadbot.core.facade_utils import DadBotFacadeUtils
        return DadBotFacadeUtils.coerce_memory_summary(value)



    def new_chat_session(self) -> Any:
        from dadbot.core.facade_utils import DadBotFacadeUtils
        return DadBotFacadeUtils.new_chat_session(self)

    # ------------------------------------------------------------------
    # Core manager properties
    # ------------------------------------------------------------------


    @property
    def memory(self) -> MemoryManager:
        """Returns the memory manager. Guaranteed non-None."""
        return self._memory_manager


    @property
    def relationship(self) -> RelationshipManager:
        """Returns the relationship manager. Guaranteed non-None."""
        return self._relationship_manager


    @property
    def mood(self) -> MoodManager:
        """Returns the mood manager. Guaranteed non-None."""
        return self._mood_manager


    @property
    def profile(self) -> ProfileRuntime:
        """Returns the profile runtime. Guaranteed non-None."""
        return self._profile_runtime

    @property
    def turn_orchestrator(self) -> Any:
        orchestrator = getattr(self, "_turn_orchestrator", None)
        if orchestrator is not None:
            return orchestrator
        services_orchestrator = getattr(self.services, "turn_orchestrator", None)
        if services_orchestrator is not None:
            self._turn_orchestrator = services_orchestrator
            return services_orchestrator
        return self._get_turn_orchestrator()

    @property
    def assistant(self) -> AssistantRuntime:
        cached = getattr(self, "_assistant_runtime", None)
        if cached is None:
            cached = AssistantRuntime(self, entrypoint=ExecutionEntry(self.execute_turn))
            object.__setattr__(self, "_assistant_runtime", cached)
        return cast(AssistantRuntime, cached)

    @property
    def phase_closure_runtime(self) -> Any:
        cached = getattr(self, "_phase_closure_runtime", None)
        if cached is None:
            from dadbot.core.phase_closure_runtime import PhaseClosureRuntime

            cached = PhaseClosureRuntime()
            object.__setattr__(self, "_phase_closure_runtime", cached)
        return cached

    @property
    def context_builder(self) -> Any:
        """On-demand ContextBuilder for prompt composition during boot and runtime.
        
        This ensures build_core_persona_prompt() is always available even if
        context_service isn't fully wired during early boot phases.
        """
        cached = getattr(self, "_context_builder", None)
        if cached is None:
            from dadbot.context import ContextBuilder
            cached = ContextBuilder(cast(Any, self))
            object.__setattr__(self, "_context_builder", cached)
        return cached

    @context_builder.setter
    def context_builder(self, value: Any) -> None:
        """Allow setting context_builder during wiring."""
        object.__setattr__(self, "_context_builder", value)

    def build_core_persona_prompt(self) -> str:
        """Delegate to context_builder for persona prompt composition."""
        return self.context_builder.build_core_persona_prompt()


    # ------------------------------------------------------------------
    # Config properties
    # ------------------------------------------------------------------

    @property
    def PROFILE(self) -> Any:
        return self._profile_runtime.profile

    @PROFILE.setter
    def PROFILE(self, value: Any) -> None:
        self._profile_runtime.profile = value



    @property
    def memory_store(self) -> dict[str, Any]:
        """Read-only projection of memory store."""
        mem = self.memory
        if mem is not None and hasattr(mem, "memory_projection"):
            return mem.memory_projection()
        return {}

    def store_memory(self, key: str, value: Any) -> None:
        mem = self.memory
        if mem is not None and hasattr(mem, "store"):
            mem.store(key, value)

    def delete_memory(self, key: str) -> None:
        mem = self.memory
        if mem is not None and hasattr(mem, "delete"):
            mem.delete(key)


    @property
    def STYLE(self) -> Any:
        return self._profile_runtime.style

    @STYLE.setter
    def STYLE(self, value: Any) -> None:
        self._profile_runtime.style = value

    # ------------------------------------------------------------------
    # Deterministic model port and UX gateway
    # (private attrs set by DadBotBootMixin)
    # ------------------------------------------------------------------

    @property
    def model_port(self) -> ModelPort:
        """Deterministic model interaction port.

        All LLM and embedding calls MUST route through this port to ensure
        replay correctness, deterministic execution, and certification validity.
        """
        if self._model_port is None:
            raise RuntimeError("Model port not initialized -- services not ready")
        return self._model_port

    @property
    def ux_gateway(self) -> TurnUxProjectionGateway:
        """UX projection gateway (post-execution assembly).

        Constraint: DadBot does NOT access state/fidelity directly.
        All reads route through cached values from previous gateway projections.
        """
        return self._ux_gateway

    # ------------------------------------------------------------------
    # Session state helpers
    # ------------------------------------------------------------------

    def reset_session_state(self) -> Any:
        """Explicit facade delegate for runtime session reset."""
        mgr = getattr(self, "_memory_manager", None)
        if mgr is not None and hasattr(mgr, 'reset_session_state'):
            result = mgr.reset_session_state()
        else:
            result = None
        # Reset CSCL session arc/pacing/coherence state for the new session
        cscl = getattr(self, "conversation_surface", None)
        if cscl is not None:
            try:
                cscl.reset_session()
            except Exception:
                pass
        return result

    def direct_reply_for_input(
        self,
        stripped_input: str,
        current_mood: str,
    ) -> str | None:
        """Explicit facade delegate for TurnService direct reply helpers.

        This avoids accidental routing to similarly-named safety helpers.
        """

        turn_service = getattr(self, 'turn_service', None)
        if turn_service and hasattr(turn_service, 'direct_reply_for_input'):
            return turn_service.direct_reply_for_input(stripped_input, current_mood)
        return None

    def build_image_analysis_prompt(
        self,
        note: str = "",
        *,
        user_input: str = "",
        attachment: dict | None = None,
    ) -> str:
        """Explicit facade delegate for multimodal image prompt assembly.

        Ensures the facade routes through the multimodal manager, even if other
        services expose the same method name.
        """

        multimodal = getattr(self, "multimodal_handler", None)
        builder = getattr(multimodal, "build_image_analysis_prompt", None)
        if callable(builder):
            return str(
                builder(
                    note=note,
                    user_input=user_input,
                    attachment=attachment,
                )
                or ""
            )
        return ""

    def _load_memory_store(self) -> dict[str, Any]:
        """Explicit compatibility delegate for memory-store loading.

        Some regression tests and operational tooling call this private helper
        directly to validate corruption recovery behaviour.
        """

        memory = getattr(self, "memory", None)
        loader = getattr(memory, "load_memory_store", None)
        if callable(loader):
            loaded = loader()
            return dict(loaded or {}) if isinstance(loaded, dict) else {}
        return {}

    def _thread_timestamp(self) -> Any:
        mgr = getattr(self, '_memory_manager', None)
        if mgr and hasattr(mgr, 'thread_timestamp'):
            return mgr.thread_timestamp()
        return None

    def _apply_thread_snapshot_unlocked(self, snapshot: Any) -> Any:
        mgr = getattr(self, '_memory_manager', None)
        if mgr and hasattr(mgr, 'apply_thread_snapshot_unlocked'):
            return mgr.apply_thread_snapshot_unlocked(snapshot)
        return None



    def record_shadow_decision(
        self,
        *,
        source: str,
        type: str,
        content_preview: str = "",
        reason: str = "",
        would_replace: bool = False,
        priority: float = 0.0,
        timestamp: float | None = None,
        metadata: dict[str, Any] | None = None,
        turn_context: Any | None = None,
    ) -> dict[str, Any]:
        """Emit a shadow-decision event via the event bus."""
        raw_type = str(type or "suggestion").strip().lower()
        allowed_types = {"override_attempt", "veto", "suggestion", "transform"}
        normalized_type = raw_type if raw_type in allowed_types else "suggestion"
        event: dict[str, Any] = {
            "source": str(source or "unknown").strip().lower() or "unknown",
            "type": normalized_type,
            "content_preview": str(content_preview or "")[:280],
            "reason": str(reason or "")[:320],
            "would_replace": bool(would_replace),
            "priority": float(priority or 0.0),
            "timestamp": float(timestamp if timestamp is not None else time.time()),
        }
        if isinstance(metadata, dict) and metadata:
            event["metadata"] = dict(metadata)
        self._event_bus.emit(event)
        return event



    def peek_shadow_decisions(self, limit: int = 64) -> list[dict[str, Any]]:
        return self._event_bus.peek(limit)



    def consume_shadow_decisions(self, limit: int = 128) -> list[dict[str, Any]]:
        return self._event_bus.consume(limit)

    def record_shadow_decision_report(self, report: dict[str, Any]) -> dict[str, Any]:
        payload = dict(report or {})
        self._last_shadow_decision_report = payload
        return payload

    def shadow_decision_report(self) -> dict[str, Any]:
        return dict(getattr(self, "_last_shadow_decision_report", {}) or {})

    @property
    def script_path(self) -> Any:
        """Resolved runtime entry script path used by persistence helpers."""
        return self.runtime_script_path()


if __name__ == "__main__":
    from dadbot.app_runtime import (
        main as run_app_main,  # type: ignore[reportUnknownVariableType]
    )

    raise SystemExit(run_app_main(dadbot_cls=DadBot, script_path=__file__))
