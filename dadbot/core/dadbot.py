"""dadbot/core/dadbot.py — DadBot public facade.

DadBot is intentionally thin.  Logic lives in dedicated managers held by
``self.services`` and in the five behaviour mixins:

    DadBotBootMixin     boot/init/shutdown lifecycle
    DadBotTurnMixin     turn execution and graph failure handling
    DadBotLlmMixin      LLM/model call forwarding
    DadBotMcpMixin      local MCP server management
    DadBotHealthMixin   health/UX state and checkpoint/replay

The remaining body of this class is pure delegation plumbing:
    * Class-level routing maps (_CONFIG_ATTR_MAP, _MANAGER_DELEGATE_CHAIN, etc.)
    * __getattr__ / __setattr__ for zero-overhead manager attribute routing
    * Explicit @property getters/setters for every registered manager
    * Config and runtime-state property aliases (PROFILE, MEMORY_STORE, ...)
    * model_port / ux_gateway properties (access private attrs set by boot mixin)
    * reset_session_state and two small turn-state helpers
"""

from __future__ import annotations

import logging
import time
from typing import Any, cast

try:
    import ollama
except ImportError:  # pragma: no cover
    ollama = None  # type: ignore[assignment]

from dadbot.assistant_runtime import AssistantRuntime
from dadbot.core.action_mixin import DadBotActionMixin
from dadbot.core.boot_mixin import DadBotBootMixin
from dadbot.core.compat_mixin import DadBotCompatMixin
from dadbot.core.convenience_mixin import DadBotConvenienceMixin
from dadbot.core.facade_compat import (
    DadBotFacadeCompat,  # noqa: F401  # type: ignore[reportUnusedImport]
)
from dadbot.core.health_mixin import DadBotHealthMixin
from dadbot.core.llm_mixin import DadBotLlmMixin
from dadbot.core.mcp_mixin import DadBotMcpMixin
from dadbot.core.turn_mixin import DadBotTurnMixin
from dadbot.core.execution_contract import ExecutionEntry
from dadbot.core.ux_projection_gateway import TurnUxProjectionGateway
from dadbot.runtime.model import ModelPort

if ollama is None:
    logging.getLogger(__name__).warning(
        "ollama package is not installed; Ollama-backed features will be unavailable. Install with: pip install ollama",
    )

logger = logging.getLogger(__name__)





class _MemoryProjectionProxy(dict):
    """Compatibility dict facade that routes writes through canonical events."""

    def __init__(self, bot: "DadBot") -> None:
        super().__init__()
        self._bot = bot

    def _snapshot(self) -> dict[str, Any]:
        memory = getattr(self._bot, "memory", None)
        if memory is None:
            return {}
        return dict(memory.memory_projection() or {})

    def __getitem__(self, key: Any) -> Any:
        return self._snapshot()[key]

    def get(self, key: Any, default: Any = None) -> Any:
        return self._snapshot().get(key, default)

    def __iter__(self):
        return iter(self._snapshot())

    def __len__(self) -> int:
        return len(self._snapshot())

    def keys(self):
        return self._snapshot().keys()

    def values(self):
        return self._snapshot().values()

    def items(self):
        return self._snapshot().items()

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, _MemoryProjectionProxy):
            return self._snapshot() == other._snapshot()
        if isinstance(other, dict):
            return self._snapshot() == other
        return self._snapshot() == other

    def __repr__(self) -> str:
        return repr(self._snapshot())

    def __setitem__(self, key: Any, value: Any) -> None:
        self._bot.mutate_memory_store(save=False, **{str(key): value})

    def __delitem__(self, key: Any) -> None:
        self._bot.mutate_memory_store(save=False, **{str(key): None})

    def update(self, *args: Any, **kwargs: Any) -> None:
        updates = dict(*args, **kwargs)
        if updates:
            self._bot.mutate_memory_store(save=False, **updates)

    def clear(self) -> None:
        self._bot.memory.clear_memory_store()


class DadBot(
    DadBotBootMixin,
    DadBotTurnMixin,
    DadBotLlmMixin,
    DadBotMcpMixin,
    DadBotHealthMixin,
    DadBotCompatMixin,
    DadBotConvenienceMixin,
    DadBotActionMixin,
):
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
    3. **Auto-delegation via __getattr__** -- unknown attribute lookups are forwarded in
       priority order through ``_MANAGER_DELEGATE_CHAIN``; first match wins.
    """

    # ------------------------------------------------------------------
    # Unified routing: combined config, runtime state, and internal maps
    # ------------------------------------------------------------------
    # Format: name -> (target_object, target_attr) for delegation

    _UNIFIED_ROUTING: dict[str, tuple[str, str]] = {
        # CONFIG mappings: route to self.config.<attr>
        "MODEL_NAME": ("config", "model_name"),
        "FALLBACK_MODELS": ("config", "fallback_models"),
        "ACTIVE_MODEL": ("config", "active_model"),
        "active_model": ("config", "active_model"),
        "ACTIVE_EMBEDDING_MODEL": ("config", "active_embedding_model"),
        "LLM_PROVIDER": ("config", "llm_provider"),
        "LLM_MODEL": ("config", "llm_model"),
        "TENANT_ID": ("config", "tenant_id"),
        "tenant_id": ("config", "tenant_id"),
        "APPEND_SIGNOFF": ("config", "append_signoff"),
        "LIGHT_MODE": ("config", "light_mode"),
        "PREFERRED_EMBEDDING_MODELS": ("config", "preferred_embedding_models"),
        "RECENT_HISTORY_WINDOW": ("config", "recent_history_window"),
        "MAX_HISTORY_MESSAGES_SCAN": ("config", "max_history_messages_scan"),
        "SUMMARY_TRIGGER_MESSAGES": ("config", "summary_trigger_messages"),
        "RELATIONSHIP_REFLECTION_INTERVAL": ("config", "relationship_reflection_interval"),
        "CONTEXT_TOKEN_BUDGET": ("config", "context_token_budget"),
        "RESERVED_RESPONSE_TOKENS": ("config", "reserved_response_tokens"),
        "APPROX_CHARS_PER_TOKEN": ("config", "approx_chars_per_token"),
        "MOOD_DETECTION_TEMPERATURE": ("config", "mood_detection_temperature"),
        "STREAM_TIMEOUT_SECONDS": ("config", "stream_timeout_seconds"),
        "STREAM_MAX_CHARS": ("config", "stream_max_chars"),
        "PROFILE_PATH": ("config", "profile_path"),
        "MEMORY_PATH": ("config", "memory_path"),
        "SEMANTIC_MEMORY_DB_PATH": ("config", "semantic_memory_db_path"),
        "GRAPH_STORE_DB_PATH": ("config", "graph_store_db_path"),
        "SESSION_LOG_DIR": ("config", "session_log_dir"),
        # RUNTIME_STATE mappings: route to self.runtime_state_manager.<attr>
        "runtime_state_container": ("runtime_state_manager", "container"),
        "history": ("runtime_state_manager", "history"),
        "session_moods": ("runtime_state_manager", "session_moods"),
        "session_summary": ("runtime_state_manager", "session_summary"),
        "session_summary_updated_at": ("runtime_state_manager", "session_summary_updated_at"),
        "session_summary_covered_messages": ("runtime_state_manager", "session_summary_covered_messages"),
        "last_relationship_reflection_turn": ("runtime_state_manager", "last_relationship_reflection_turn"),
        "_pending_daily_checkin_context": ("runtime_state_manager", "pending_daily_checkin_context"),
        "_active_tool_observation_context": ("runtime_state_manager", "active_tool_observation_context"),
        "_last_planner_debug": ("runtime_state_manager", "planner_debug"),
        "chat_threads": ("runtime_state_manager", "chat_threads"),
        "active_thread_id": ("runtime_state_manager", "active_thread_id"),
        "thread_snapshots": ("runtime_state_manager", "thread_snapshots"),
        # INTERNAL_RUNTIME mappings: route to self._internal_runtime.<attr>
        "_prompt_guard_stats": ("_internal_runtime", "prompt_guard_stats"),
        "_last_memory_context_stats": ("_internal_runtime", "last_memory_context_stats"),
        "_last_output_moderation": ("_internal_runtime", "last_output_moderation"),
        "_last_reply_supervisor": ("_internal_runtime", "last_reply_supervisor"),
        "_last_turn_pipeline": ("_internal_runtime", "last_turn_pipeline"),
        "_last_turn_health_state": ("_internal_runtime", "last_turn_health_state"),
        "_last_turn_ux_feedback": ("_internal_runtime", "last_turn_ux_feedback"),
        "_shadow_decision_bus": ("_internal_runtime", "shadow_decision_bus"),
        "_last_shadow_decision_report": ("_internal_runtime", "last_shadow_decision_report"),
        "_background_task_ids": ("_internal_runtime", "background_task_ids"),
    }

    # ------------------------------------------------------------------
    # Fallback manager delegation chain
    # ------------------------------------------------------------------

    _MANAGER_DELEGATE_CHAIN: tuple[str, ...] = (
        "health_manager", "tts_manager", "avatar_manager", "calendar_manager",
        "email_manager", "runtime_storage", "profile_runtime", "memory_manager",
        "memory_coordinator", "long_term_signals", "relationship_manager",
        "runtime_state_manager", "status_reporting",  # legacy
        "maintenance_scheduler", "conversation_persistence", "session_summary_manager",
        "turn_service", "memory_query", "memory_commands", "safety_support",
        "profile_context", "reply_supervisor", "reply_finalization", "multimodal_handler",
        "model_runtime", "agentic_handler", "tool_registry", "context_service",
        "tone_context", "personality_service", "conversation_surface", "prompt_assembly",
        "runtime_client", "runtime_orchestration", "runtime_interface", "mood_manager",
    )

    _DEPRECATED_FACADE_ALIASES: dict[str, str] = {
        "detect_mood": "mood_manager.detect",
        "detect_mood_async": "mood_manager.detect_async",
        "fastpath_detect_mood": "mood_manager.fastpath_detect",
        "finalize_reply": "reply_finalization.append_signoff",
        "prepare_final_reply": "reply_finalization.finalize",
        "prepare_final_reply_async": "reply_finalization.finalize_async",
        "relationship_emotional_momentum": "relationship.emotional_momentum",
        "top_relationship_topics": "relationship.top_topics",
        "relationship_hypotheses": "relationship.hypotheses",
        "relationship_snapshot": "relationship.snapshot",
        "update_relationship_state": "relationship.current_state",
        "apply_relationship_feedback": "relationship.apply_feedback",
        "build_relationship_reflection_prompt": "relationship.build_reflection_prompt",
        "reflect_relationship_state": "relationship.current_state",
        "internal_state_snapshot": "internal_state_manager.snapshot",
        "reflect_internal_state": "internal_state_manager.reflect_after_turn",
        "record_user_turn_state": "turn_service.record_user_turn_state",
        "evolved_persona_traits": "profile_runtime.evolved_persona_traits",
        "active_persona_traits": "profile_runtime.active_persona_traits",
        "effective_behavior_rules": "profile_runtime.effective_behavior_rules",
        "living_dad_snapshot": "profile_runtime.living_dad_snapshot",
    }

    # ------------------------------------------------------------------
    # Unified attribute routing
    # ------------------------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        """Route attribute lookups via unified routing map or manager delegation chain."""
        if name.startswith("__"):
            raise AttributeError(name)

        # Unified routing: check config, runtime_state, internal_runtime
        route = self.__class__._UNIFIED_ROUTING.get(name)
        if route is not None:
            target_obj_name, target_attr = route
            try:
                target = object.__getattribute__(self, target_obj_name)
                return getattr(target, target_attr)
            except AttributeError:
                pass

        # Services provider fallback
        try:
            services = object.__getattribute__(self, "services")
            provider = services.get_provider(name)
            if provider is not None:
                compat = self.__dict__.get("_facade_compat")
                if compat is not None:
                    compat.warn_if_deprecated(name)
                return getattr(provider, name)
        except AttributeError:
            pass

        # Manager delegation chain fallback
        # Managers are stored in services, so we access them through there
        _sentinel = object()
        try:
            services = object.__getattribute__(self, "services")
            for _mgr_attr in self.__class__._MANAGER_DELEGATE_CHAIN:
                try:
                    _mgr = getattr(services, _mgr_attr, _sentinel)
                    if _mgr is not _sentinel:
                        _val = getattr(_mgr, name, _sentinel)
                        if _val is not _sentinel:
                            return _val
                except AttributeError:
                    continue
        except AttributeError:
            pass
        
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name!r}'",
        )

    def __setattr__(self, name: str, value: Any) -> None:
        """Route mutations via unified routing map."""
        # Unified routing: check config, runtime_state, internal_runtime
        route = self.__class__._UNIFIED_ROUTING.get(name)
        if route is not None:
            target_obj_name, target_attr = route
            try:
                target = object.__getattribute__(self, target_obj_name)
                setattr(target, target_attr, value)
                return
            except AttributeError:
                pass

        object.__setattr__(self, name, value)

    def _get_explicit_manager(self, name: str) -> Any:
        sentinel = object()
        value = self.__dict__.get(f"_{name}", sentinel)
        if value is not sentinel:
            return value
        services = self.__dict__.get("services")
        if services is not None:
            value = getattr(services, name, sentinel)
            if value is not sentinel:
                return value
        raise AttributeError(name)

    def _set_explicit_manager(self, name: str, value: Any) -> None:
        object.__setattr__(self, f"_{name}", value)

    def _resolve_dependency(self, name: str, factory: Any) -> Any:
        """Resolve a runtime dependency from the optional injection registry."""
        registry = getattr(self, "_dependency_registry", None)
        candidate = None
        if registry is not None and hasattr(registry, "get"):
            candidate = registry.get(name)
        if callable(candidate):
            candidate = candidate()
        if candidate is not None:
            return candidate
        return factory()

    # ------------------------------------------------------------------
    # Core manager properties
    # ------------------------------------------------------------------

    @property
    def memory(self) -> Any:
        return getattr(self.services, "memory_manager", None)

    @property
    def relationship(self) -> Any:
        return getattr(self.services, "relationship_manager", None)

    @property
    def mood(self) -> Any:
        return getattr(self.services, "mood_manager", None)

    @property
    def profile(self) -> Any:
        return getattr(self.services, "profile_runtime", None)

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
            cached = ContextBuilder(self)
            object.__setattr__(self, "_context_builder", cached)
        return cached

    @context_builder.setter
    def context_builder(self, value: Any) -> None:
        """Allow setting context_builder during wiring."""
        object.__setattr__(self, "_context_builder", value)

    def build_core_persona_prompt(self) -> str:
        """Delegate to context_builder for persona prompt composition."""
        return self.context_builder.build_core_persona_prompt()

    @property
    def memory_query(self) -> Any:
        """Alias for memory_manager."""
        return self._get_explicit_manager("memory_manager")

    @memory_query.setter
    def memory_query(self, value: Any) -> None:
        self._set_explicit_manager("memory_manager", value)

    # ------------------------------------------------------------------
    # Config properties
    # ------------------------------------------------------------------

    @property
    def PROFILE(self):
        return self.profile_runtime.profile

    @PROFILE.setter
    def PROFILE(self, value: Any):
        self.profile_runtime.profile = value

    @property
    def MEMORY_STORE(self):
        proxy = getattr(self, "_memory_store_proxy", None)
        if proxy is None:
            proxy = _MemoryProjectionProxy(self)
            self._memory_store_proxy = proxy
        return proxy

    @MEMORY_STORE.setter
    def MEMORY_STORE(self, value: Any):
        # Compatibility surface remains, but reset must still flow through
        # canonical event routing to preserve deterministic state transitions.
        _mem = self.memory
        if _mem is not None:
            store: dict[str, Any] = cast("dict[str, Any]", value) if isinstance(value, dict) else {}
            replace_canonical = getattr(_mem._storage, "replace_projection_via_canonical_event", None)
            if callable(replace_canonical):
                replace_canonical(store, save=False)
                return
            replace_projection = getattr(_mem, "_set_memory_projection_cache", None)
            if callable(replace_projection):
                replace_projection(store)

    @property
    def STYLE(self):
        return self.profile_runtime.style

    @STYLE.setter
    def STYLE(self, value: Any):
        self.profile_runtime.style = value

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
        mgr = getattr(self, "runtime_state_manager", None)
        if mgr is not None:
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

    def _thread_timestamp(self) -> Any:
        return self.runtime_state_manager.thread_timestamp()

    def _apply_thread_snapshot_unlocked(self, snapshot: Any) -> Any:
        return self.runtime_state_manager.apply_thread_snapshot_unlocked(snapshot)

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
        """Record a normalized observational shadow-decision event.

        This event stream is strictly telemetry and does not participate in
        response selection authority.
        """
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

        bus = list(getattr(self, "_shadow_decision_bus", []) or [])
        bus.append(event)
        # Keep bounded in memory for diagnostics without unbounded growth.
        self._shadow_decision_bus = bus[-256:]

        context_meta = getattr(turn_context, "metadata", None)
        if isinstance(context_meta, dict):
            context_stream = list(context_meta.get("shadow_decision_bus") or [])
            context_stream.append(dict(event))
            context_meta["shadow_decision_bus"] = context_stream[-128:]

        return event

    def peek_shadow_decisions(self, limit: int = 64) -> list[dict[str, Any]]:
        stream = list(getattr(self, "_shadow_decision_bus", []) or [])
        max_items = max(1, int(limit or 1))
        return stream[-max_items:]

    def consume_shadow_decisions(self, limit: int = 128) -> list[dict[str, Any]]:
        stream = list(getattr(self, "_shadow_decision_bus", []) or [])
        max_items = max(1, int(limit or 1))
        consumed = stream[-max_items:]
        self._shadow_decision_bus = []
        return consumed

    def record_shadow_decision_report(self, report: dict[str, Any]) -> dict[str, Any]:
        payload = dict(report or {})
        self._last_shadow_decision_report = payload
        return payload

    def shadow_decision_report(self) -> dict[str, Any]:
        return dict(getattr(self, "_last_shadow_decision_report", {}) or {})

    @property
    def script_path(self):
        """Resolved runtime entry script path used by persistence helpers."""
        return self.runtime_script_path()


if __name__ == "__main__":
    from dadbot.app_runtime import (
        main as run_app_main,  # type: ignore[reportUnknownVariableType]
    )

    raise SystemExit(run_app_main(dadbot_cls=DadBot, script_path=__file__))
