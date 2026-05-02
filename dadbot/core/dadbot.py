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
from typing import Any, cast

try:
    import ollama
except ImportError:  # pragma: no cover
    ollama = None  # type: ignore[assignment]

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
    # Auto-delegation registry
    # ------------------------------------------------------------------

    _LEGACY_MANAGER_DELEGATE_CHAIN: tuple[str, ...] = (
        "health_manager",
        "tts_manager",
        "avatar_manager",
        "calendar_manager",
        "email_manager",
        "runtime_storage",
        "profile_runtime",
        "memory_manager",
        "memory_coordinator",
        "long_term_signals",
        "relationship_manager",
        "runtime_state_manager",
        "status_reporting",
    )

    _SERVICE_MANAGER_DELEGATE_CHAIN: tuple[str, ...] = (
        "maintenance_scheduler",
        "conversation_persistence",
        "session_summary_manager",
        "turn_service",
        "memory_query",
        "memory_commands",
        "safety_support",
        "profile_context",
        "reply_supervisor",
        "reply_finalization",
        "multimodal_handler",
        "model_runtime",
        "agentic_handler",
        "tool_registry",
        "context_service",
        "tone_context",
        "personality_service",
        "prompt_assembly",
        "runtime_client",
        "runtime_orchestration",
        "runtime_interface",
        "mood_manager",
    )

    _MANAGER_DELEGATE_CHAIN: tuple[str, ...] = _LEGACY_MANAGER_DELEGATE_CHAIN + _SERVICE_MANAGER_DELEGATE_CHAIN

    _CONFIG_ATTR_MAP: dict[str, str] = {
        "MODEL_NAME": "model_name",
        "FALLBACK_MODELS": "fallback_models",
        "ACTIVE_MODEL": "active_model",
        "active_model": "active_model",
        "ACTIVE_EMBEDDING_MODEL": "active_embedding_model",
        "LLM_PROVIDER": "llm_provider",
        "LLM_MODEL": "llm_model",
        "TENANT_ID": "tenant_id",
        "tenant_id": "tenant_id",
        "APPEND_SIGNOFF": "append_signoff",
        "LIGHT_MODE": "light_mode",
        "PREFERRED_EMBEDDING_MODELS": "preferred_embedding_models",
        "RECENT_HISTORY_WINDOW": "recent_history_window",
        "MAX_HISTORY_MESSAGES_SCAN": "max_history_messages_scan",
        "SUMMARY_TRIGGER_MESSAGES": "summary_trigger_messages",
        "RELATIONSHIP_REFLECTION_INTERVAL": "relationship_reflection_interval",
        "CONTEXT_TOKEN_BUDGET": "context_token_budget",
        "RESERVED_RESPONSE_TOKENS": "reserved_response_tokens",
        "APPROX_CHARS_PER_TOKEN": "approx_chars_per_token",
        "MOOD_DETECTION_TEMPERATURE": "mood_detection_temperature",
        "STREAM_TIMEOUT_SECONDS": "stream_timeout_seconds",
        "STREAM_MAX_CHARS": "stream_max_chars",
        "PROFILE_PATH": "profile_path",
        "MEMORY_PATH": "memory_path",
        "SEMANTIC_MEMORY_DB_PATH": "semantic_memory_db_path",
        "GRAPH_STORE_DB_PATH": "graph_store_db_path",
        "SESSION_LOG_DIR": "session_log_dir",
    }

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

    _RUNTIME_STATE_ATTR_MAP: dict[str, str] = {
        "runtime_state_container": "container",
        "history": "history",
        "session_moods": "session_moods",
        "session_summary": "session_summary",
        "session_summary_updated_at": "session_summary_updated_at",
        "session_summary_covered_messages": "session_summary_covered_messages",
        "last_relationship_reflection_turn": "last_relationship_reflection_turn",
        "_pending_daily_checkin_context": "pending_daily_checkin_context",
        "_active_tool_observation_context": "active_tool_observation_context",
        "_last_planner_debug": "planner_debug",
        "chat_threads": "chat_threads",
        "active_thread_id": "active_thread_id",
        "thread_snapshots": "thread_snapshots",
    }

    _INTERNAL_RUNTIME_ATTR_MAP: dict[str, str] = {
        "_prompt_guard_stats": "prompt_guard_stats",
        "_last_memory_context_stats": "last_memory_context_stats",
        "_last_output_moderation": "last_output_moderation",
        "_last_reply_supervisor": "last_reply_supervisor",
        "_last_turn_pipeline": "last_turn_pipeline",
        "_last_turn_health_state": "last_turn_health_state",
        "_last_turn_ux_feedback": "last_turn_ux_feedback",
        "_background_task_ids": "background_task_ids",
    }

    # ------------------------------------------------------------------
    # Attribute routing
    # ------------------------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        """Route unknown attribute lookups to registered manager objects.

        After __init__ completes, lookups hit the O(1) provider map exposed by
        self.services.  During the early-init window the chain is walked
        directly so that manager construction can reference each other safely.

        Dunder names (__foo__) are never delegated.
        """
        if name.startswith("__"):
            raise AttributeError(name)

        config_attr = self.__class__._CONFIG_ATTR_MAP.get(name)
        if config_attr is not None:
            try:
                config = object.__getattribute__(self, "config")
                return getattr(config, config_attr)
            except AttributeError:
                pass

        runtime_state_attr = self.__class__._RUNTIME_STATE_ATTR_MAP.get(name)
        if runtime_state_attr is not None:
            try:
                runtime_state_manager = object.__getattribute__(
                    self,
                    "runtime_state_manager",
                )
                return getattr(runtime_state_manager, runtime_state_attr)
            except AttributeError:
                pass

        internal_runtime_attr = self.__class__._INTERNAL_RUNTIME_ATTR_MAP.get(name)
        if internal_runtime_attr is not None:
            try:
                internal_runtime = object.__getattribute__(self, "_internal_runtime")
                return getattr(internal_runtime, internal_runtime_attr)
            except AttributeError:
                pass

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

        _sentinel = object()
        for _mgr_attr in self.__class__._MANAGER_DELEGATE_CHAIN:
            try:
                _mgr = object.__getattribute__(self, _mgr_attr)
            except AttributeError:
                continue
            _val = getattr(_mgr, name, _sentinel)
            if _val is not _sentinel:
                return _val
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name!r}'",
        )

    def __setattr__(self, name: str, value: Any) -> None:
        """Route mutations of config-mirrored names to self.config."""
        config_attr = self.__class__._CONFIG_ATTR_MAP.get(name)
        if config_attr is not None:
            try:
                config = object.__getattribute__(self, "config")
                setattr(config, config_attr, value)
                return
            except AttributeError:
                pass

        runtime_state_attr = self.__class__._RUNTIME_STATE_ATTR_MAP.get(name)
        if runtime_state_attr is not None:
            try:
                runtime_state_manager = object.__getattribute__(
                    self,
                    "runtime_state_manager",
                )
                setattr(runtime_state_manager, runtime_state_attr, value)
                return
            except AttributeError:
                pass

        internal_runtime_attr = self.__class__._INTERNAL_RUNTIME_ATTR_MAP.get(name)
        if internal_runtime_attr is not None:
            try:
                internal_runtime = object.__getattribute__(self, "_internal_runtime")
                setattr(internal_runtime, internal_runtime_attr, value)
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
        return self.services.turn_orchestrator

    # ------------------------------------------------------------------
    # Explicit manager properties (getter + setter)
    # ------------------------------------------------------------------

    @property
    def runtime_storage(self):
        return self._get_explicit_manager("runtime_storage")

    @runtime_storage.setter
    def runtime_storage(self, value: Any):
        self._set_explicit_manager("runtime_storage", value)

    @property
    def profile_runtime(self):
        return self._get_explicit_manager("profile_runtime")

    @profile_runtime.setter
    def profile_runtime(self, value: Any):
        self._set_explicit_manager("profile_runtime", value)

    @property
    def mood_manager(self):
        return self._get_explicit_manager("mood_manager")

    @mood_manager.setter
    def mood_manager(self, value: Any):
        self._set_explicit_manager("mood_manager", value)

    @property
    def turn_service(self):
        return self._get_explicit_manager("turn_service")

    @turn_service.setter
    def turn_service(self, value: Any):
        self._set_explicit_manager("turn_service", value)

    @property
    def reply_finalization(self):
        return self._get_explicit_manager("reply_finalization")

    @reply_finalization.setter
    def reply_finalization(self, value: Any):
        self._set_explicit_manager("reply_finalization", value)

    @property
    def runtime_interface(self):
        return self._get_explicit_manager("runtime_interface")

    @runtime_interface.setter
    def runtime_interface(self, value: Any):
        self._set_explicit_manager("runtime_interface", value)

    @property
    def status_reporting(self):
        return self._get_explicit_manager("status_reporting")

    @status_reporting.setter
    def status_reporting(self, value: Any):
        self._set_explicit_manager("status_reporting", value)

    @property
    def model_runtime(self):
        return self._get_explicit_manager("model_runtime")

    @model_runtime.setter
    def model_runtime(self, value: Any):
        self._set_explicit_manager("model_runtime", value)

    @property
    def runtime_client(self):
        return self._get_explicit_manager("runtime_client")

    @runtime_client.setter
    def runtime_client(self, value: Any):
        self._set_explicit_manager("runtime_client", value)

    @property
    def maintenance_scheduler(self):
        return self._get_explicit_manager("maintenance_scheduler")

    @maintenance_scheduler.setter
    def maintenance_scheduler(self, value: Any):
        self._set_explicit_manager("maintenance_scheduler", value)

    @property
    def health_manager(self):
        return self._get_explicit_manager("health_manager")

    @health_manager.setter
    def health_manager(self, value: Any):
        self._set_explicit_manager("health_manager", value)

    @property
    def internal_state_manager(self):
        return self._get_explicit_manager("internal_state_manager")

    @internal_state_manager.setter
    def internal_state_manager(self, value: Any):
        self._set_explicit_manager("internal_state_manager", value)

    @property
    def runtime_state_manager(self):
        return self._get_explicit_manager("runtime_state_manager")

    @runtime_state_manager.setter
    def runtime_state_manager(self, value: Any):
        self._set_explicit_manager("runtime_state_manager", value)

    @property
    def prompt_assembly(self):
        return self._get_explicit_manager("prompt_assembly")

    @prompt_assembly.setter
    def prompt_assembly(self, value: Any):
        self._set_explicit_manager("prompt_assembly", value)

    @property
    def context_service(self):
        return self._get_explicit_manager("context_service")

    @context_service.setter
    def context_service(self, value: Any):
        self._set_explicit_manager("context_service", value)

    @property
    def context_builder(self):
        # Compatibility alias during context_builder collapse.
        return self._get_explicit_manager("context_service")

    @context_builder.setter
    def context_builder(self, value: Any):
        self._set_explicit_manager("context_service", value)

    @property
    def tone_context(self):
        return self._get_explicit_manager("tone_context")

    @tone_context.setter
    def tone_context(self, value: Any):
        self._set_explicit_manager("tone_context", value)

    @property
    def memory_query(self):
        # Compatibility alias during memory_query collapse.
        return self._get_explicit_manager("memory_manager")

    @memory_query.setter
    def memory_query(self, value: Any):
        self._set_explicit_manager("memory_manager", value)

    @property
    def memory_commands(self):
        return self._get_explicit_manager("memory_commands")

    @memory_commands.setter
    def memory_commands(self, value: Any):
        self._set_explicit_manager("memory_commands", value)

    @property
    def memory_coordinator(self):
        return self._get_explicit_manager("memory_coordinator")

    @memory_coordinator.setter
    def memory_coordinator(self, value: Any):
        self._set_explicit_manager("memory_coordinator", value)

    @property
    def long_term_signals(self):
        return self._get_explicit_manager("long_term_signals")

    @long_term_signals.setter
    def long_term_signals(self, value: Any):
        self._set_explicit_manager("long_term_signals", value)

    @property
    def safety_support(self):
        return self._get_explicit_manager("safety_support")

    @safety_support.setter
    def safety_support(self, value: Any):
        self._set_explicit_manager("safety_support", value)

    @property
    def reply_supervisor(self):
        return self._get_explicit_manager("reply_supervisor")

    @reply_supervisor.setter
    def reply_supervisor(self, value: Any):
        self._set_explicit_manager("reply_supervisor", value)

    @property
    def multimodal_handler(self):
        return self._get_explicit_manager("multimodal_handler")

    @multimodal_handler.setter
    def multimodal_handler(self, value: Any):
        self._set_explicit_manager("multimodal_handler", value)

    @property
    def runtime_orchestration(self):
        return self._get_explicit_manager("runtime_orchestration")

    @runtime_orchestration.setter
    def runtime_orchestration(self, value: Any):
        self._set_explicit_manager("runtime_orchestration", value)

    @property
    def session_summary_manager(self):
        return self._get_explicit_manager("session_summary_manager")

    @session_summary_manager.setter
    def session_summary_manager(self, value: Any):
        self._set_explicit_manager("session_summary_manager", value)

    @property
    def tool_registry(self):
        return self._get_explicit_manager("tool_registry")

    @tool_registry.setter
    def tool_registry(self, value: Any):
        self._set_explicit_manager("tool_registry", value)

    @property
    def agentic_handler(self):
        return self._get_explicit_manager("agentic_handler")

    @agentic_handler.setter
    def agentic_handler(self, value: Any):
        self._set_explicit_manager("agentic_handler", value)

    @property
    def conversation_persistence(self):
        return self._get_explicit_manager("conversation_persistence")

    @conversation_persistence.setter
    def conversation_persistence(self, value: Any):
        self._set_explicit_manager("conversation_persistence", value)

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
        return self.memory.memory_store

    @MEMORY_STORE.setter
    def MEMORY_STORE(self, value: Any):
        # Preserve reference identity for callers that intentionally pass a
        # prebuilt in-memory store (used by service-split compatibility tests).
        _mem = self.memory
        if _mem is not None:
            store: dict[str, Any] = cast("dict[str, Any]", value) if isinstance(value, dict) else {}
            replace_unchecked = getattr(_mem, "_replace_memory_store_unchecked", None)
            if callable(replace_unchecked):
                replace_unchecked(store)
            else:
                _mem.memory_store = store

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
            return mgr.reset_session_state()
        return None

    def _thread_timestamp(self) -> Any:
        return self.runtime_state_manager.thread_timestamp()

    def _apply_thread_snapshot_unlocked(self, snapshot: Any) -> Any:
        return self.runtime_state_manager.apply_thread_snapshot_unlocked(snapshot)

    @property
    def script_path(self):
        """Resolved runtime entry script path used by persistence helpers."""
        return self.runtime_script_path()


if __name__ == "__main__":
    from dadbot.app_runtime import (
        main as run_app_main,  # type: ignore[reportUnknownVariableType]
    )

    raise SystemExit(run_app_main(dadbot_cls=DadBot, script_path=__file__))
