from __future__ import annotations

from typing import Any

from dadbot.config_schema import ConfigSchema
from dadbot.context import ContextBuilder
from dadbot.core.graph import TurnGraph
from dadbot.infrastructure.llm import OllamaModelAdapter
from dadbot.infrastructure.storage import FileSystemAdapter
from dadbot.infrastructure.telemetry import Logger
from dadbot.managers.conversation_persistence import ConversationPersistenceManager
from dadbot.managers.long_term import LongTermSignalsManager
from dadbot.managers.maintenance import MaintenanceScheduler
from dadbot.managers.memory_commands import MemoryCommandManager as PackagedMemoryCommandManager
from dadbot.managers.memory_coordination import MemoryCoordinator as PackagedMemoryCoordinator
from dadbot.managers.memory_query import MemoryQueryManager as PackagedMemoryQueryManager
from dadbot.managers.multimodal import MultimodalManager as PackagedMultimodalManager
from dadbot.managers.profile_runtime import ProfileRuntimeManager as PackagedProfileRuntimeManager
from dadbot.managers.prompt_assembly import PromptAssemblyManager as PackagedPromptAssemblyManager
from dadbot.managers.internal_state import InternalStateManager as PackagedInternalStateManager
from dadbot.managers.reply_finalization import ReplyFinalizationManager as PackagedReplyFinalizationManager
from dadbot.managers.reply_supervisor import ReplySupervisorManager as PackagedReplySupervisorManager
from dadbot.managers.runtime_client import RuntimeClientManager as PackagedRuntimeClientManager
from dadbot.managers.runtime_interface import RuntimeInterfaceManager as PackagedRuntimeInterfaceManager
from dadbot.managers.runtime_model import RuntimeModelManager as PackagedRuntimeModelManager
from dadbot.managers.runtime_orchestration import RuntimeOrchestrationManager as PackagedRuntimeOrchestrationManager
from dadbot.managers.runtime_storage import RuntimeStorageManager as PackagedRuntimeStorageManager
from dadbot.managers.safety import SafetySupportManager as PackagedSafetySupportManager
from dadbot.managers.status_reporting import StatusReportingManager as PackagedStatusReportingManager
from dadbot.memory.manager import MemoryManager as PackagedMemoryManager
from dadbot.mood import MoodManager as PackagedMoodManager
from dadbot.profile import ProfileContextManager as PackagedProfileContextManager
from dadbot.relationship import RelationshipManager as PackagedRelationshipManager
from dadbot.managers.health import RuntimeHealthManager as PackagedRuntimeHealthManager
from dadbot.managers.tts import TTSManager as PackagedTTSManager
from dadbot.managers.avatar import AvatarManager as PackagedAvatarManager
from dadbot.managers.calendar import CalendarManager as PackagedCalendarManager
from dadbot.managers.email import EmailManager as PackagedEmailManager
from dadbot.managers.session_summary import SessionSummaryManager as PackagedSessionSummaryManager
from dadbot.agentic import AgenticHandler as PackagedAgenticHandler, ToolRegistry as PackagedToolRegistry
from dadbot.tone import ToneContextBuilder as PackagedToneContextBuilder
from dadbot.state import RuntimeStateManager as PackagedRuntimeStateManager
from dadbot.services.agent_service import AgentService
from dadbot.services.context_service import ContextService
from dadbot.services.maintenance_service import MaintenanceService
from dadbot.services.persistence import PersistenceService
from dadbot.services.runtime_service import RuntimeService
from dadbot.services.safety_service import SafetyService
from dadbot.services.turn_service import TurnService
from dadbot.core.facade_validator import validate_dadbot_facade


_registry_logger = __import__("logging").getLogger(__name__)


class ServiceRegistry:
    """Dependency-injection registry for DadBot manager instances.

    Maintains two parallel structures:
    - ``_services``     â€” name â†’ manager instance (for direct lookup by manager name)
    - ``_provider_map`` â€” public attr name â†’ manager instance (for O(1) delegation)

    Registration order matters: the *first* manager registered that exposes a
    given attribute name wins, matching the priority order of
    ``DadBot._MANAGER_DELEGATE_CHAIN``.
    """

    def __init__(self):
        self._services: dict[str, Any] = {}
        self._provider_map: dict[str, Any] = {}

    def register(self, name: str, instance: Any) -> Any:
        self._services[name] = instance
        # Index every public attribute of this service into the provider map.
        # Only write if not already claimed (first-registered manager wins).
        for attr in dir(instance):
            if not attr.startswith("_") and attr not in self._provider_map:
                self._provider_map[attr] = instance
        return instance

    def get(self, name: str) -> Any:
        if name not in self._services:
            raise KeyError(f"Service '{name}' is not registered")
        return self._services[name]

    def get_provider(self, name: str) -> Any | None:
        """Return the manager instance that owns attribute *name*, or ``None``.

        O(1) â€” backed by the pre-built ``_provider_map`` index.  The caller is
        expected to do ``getattr(provider, name)`` to obtain the actual value.
        """
        return self._provider_map.get(name)

    def shutdown_all(self) -> None:
        """Gracefully shut down every registered service that exposes ``shutdown()``."""
        for svc_name, instance in list(self._services.items()):
            shutdown_fn = getattr(instance, "shutdown", None)
            if callable(shutdown_fn):
                try:
                    shutdown_fn()
                except Exception as exc:
                    _registry_logger.warning("Error shutting down service %r: %s", svc_name, exc)

    @classmethod
    def boot(cls, config_path: str = "config.yaml", *, bot: Any | None = None) -> "ServiceRegistry":
        registry = cls()
        config = ConfigSchema.from_file(config_path)
        registry.register("config", config)

        telemetry = registry.register("telemetry", Logger("dadbot.orchestration"))
        registry.register("storage_backend", FileSystemAdapter())
        registry.register("model_adapter", OllamaModelAdapter())

        if bot is None:
            telemetry.warning("Booted registry without bot runtime; service wrappers disabled")
            registry.register("turn_graph", TurnGraph(registry))
            return registry

        persistence_manager = ConversationPersistenceManager(bot)
        context_builder = ContextBuilder(bot)
        maintenance_manager = MaintenanceScheduler(bot)
        long_term_manager = LongTermSignalsManager(bot)

        # turn_service is already wired by wire_runtime_managers before the registry
        # is first booted (lazy, on the first graph turn). Passing it here enables
        # PersistenceService.finalize_turn to commit history, maintenance, reflection,
        # and health snapshot atomically at the end of every graph turn.
        turn_service = bot.turn_service

        # Semantic index for RAG: surfaces relevant long-term memories in ContextService
        # rather than only the most-recent window, avoiding context bloat as memory grows.
        semantic_index = None
        semantic_db_path = getattr(bot, "SEMANTIC_MEMORY_DB_PATH", None)
        if semantic_db_path is not None:
            try:
                from dadbot_system.semantic_index import SQLiteSemanticIndex
                semantic_index = SQLiteSemanticIndex(bot, semantic_db_path)
                semantic_index.ensure_storage()
            except Exception:
                pass  # Non-fatal: ContextService degrades to base context without RAG

        registry.register("persistence_service", PersistenceService(persistence_manager, turn_service=turn_service))
        registry.register("context_service", ContextService(context_builder, bot.memory_manager, semantic_index=semantic_index))
        registry.register("runtime_service", RuntimeService(bot, registry.get("model_adapter")))
        registry.register("maintenance_service", MaintenanceService(maintenance_manager, long_term_manager))
        registry.register("agent_service", AgentService(bot))
        registry.register("safety_service", SafetyService(bot))

        # Canonical aliases used by DadBotOrchestrator node lookup
        registry.register("health", registry.get("maintenance_service"))
        registry.register("memory", registry.get("context_service"))
        registry.register("llm", registry.get("agent_service"))
        registry.register("safety", registry.get("safety_service"))
        registry.register("storage", registry.get("persistence_service"))
        reflection_service = getattr(bot, "internal_state_manager", None) or registry.get("persistence_service")
        registry.register("reflection", reflection_service)
        registry.register("turn_graph", TurnGraph(registry))

        telemetry.info("ServiceRegistry boot complete", services=list(registry._services.keys()))
        return registry


def boot_registry(config_path: str = "config.yaml", *, bot: Any | None = None) -> ServiceRegistry:
    return ServiceRegistry.boot(config_path=config_path, bot=bot)


def wire_bootstrap_managers(bot: Any) -> None:
    """Attach managers required before profile/memory hydration."""
    bot.runtime_storage = bot._resolve_dependency("runtime_storage", lambda: PackagedRuntimeStorageManager(bot.bot_context))
    bot.profile_runtime = bot._resolve_dependency("profile_runtime", lambda: PackagedProfileRuntimeManager(bot))
    bot.memory_manager = bot._resolve_dependency("memory_manager", lambda: PackagedMemoryManager(bot.bot_context))


def wire_runtime_managers(bot: Any) -> None:
    """Attach all turn/runtime managers after profile/memory hydration."""

    bot.long_term_signals = bot._resolve_dependency("long_term_signals", lambda: LongTermSignalsManager(bot))
    bot.memory_query = bot._resolve_dependency("memory_query", lambda: PackagedMemoryQueryManager(bot.bot_context))
    bot.memory_commands = bot._resolve_dependency("memory_commands", lambda: PackagedMemoryCommandManager(bot.bot_context))
    bot.safety_support = bot._resolve_dependency("safety_support", lambda: PackagedSafetySupportManager(bot))
    bot.profile_context = bot._resolve_dependency("profile_context", lambda: PackagedProfileContextManager(bot))
    bot.context_builder = bot._resolve_dependency("context_builder", lambda: ContextBuilder(bot))
    bot.tone_context = bot._resolve_dependency("tone_context", lambda: PackagedToneContextBuilder(bot.bot_context))
    bot.mood_manager = bot._resolve_dependency("mood_manager", lambda: PackagedMoodManager(bot))
    bot.relationship_manager = bot._resolve_dependency("relationship_manager", lambda: PackagedRelationshipManager(bot.bot_context))
    bot.internal_state_manager = bot._resolve_dependency("internal_state_manager", lambda: PackagedInternalStateManager(bot))

    bot.prompt_assembly = bot._resolve_dependency("prompt_assembly", lambda: PackagedPromptAssemblyManager(bot.bot_context))
    bot.prompt_composer = bot.prompt_assembly
    bot.multimodal_handler = bot._resolve_dependency("multimodal_handler", lambda: PackagedMultimodalManager(bot.bot_context))
    bot.turn_service = bot._resolve_dependency(
        "turn_service",
        lambda: TurnService(bot.bot_context),
    )
    bot.reply_generation = bot.turn_service.reply_generation
    bot.reply_supervisor = bot._resolve_dependency("reply_supervisor", lambda: PackagedReplySupervisorManager(bot.bot_context))
    bot.reply_finalization = bot._resolve_dependency("reply_finalization", lambda: PackagedReplyFinalizationManager(bot))
    bot.conversation_persistence = bot._resolve_dependency("conversation_persistence", lambda: ConversationPersistenceManager(bot))
    bot.runtime_orchestration = bot._resolve_dependency("runtime_orchestration", lambda: PackagedRuntimeOrchestrationManager(bot.bot_context))
    bot.status_reporting = bot._resolve_dependency("status_reporting", lambda: PackagedStatusReportingManager(bot.bot_context))
    bot.maintenance_scheduler = bot._resolve_dependency("maintenance_scheduler", lambda: MaintenanceScheduler(bot))
    bot.runtime_interface = bot._resolve_dependency("runtime_interface", lambda: PackagedRuntimeInterfaceManager(bot.bot_context))
    bot.tool_registry = bot._resolve_dependency("tool_registry", lambda: PackagedToolRegistry(bot))
    bot.agentic_handler = bot._resolve_dependency("agentic_handler", lambda: PackagedAgenticHandler(bot, bot.tool_registry))
    bot.memory_coordinator = bot._resolve_dependency("memory_coordinator", lambda: PackagedMemoryCoordinator(bot))

    bot.health_manager = bot._resolve_dependency("health_manager", lambda: PackagedRuntimeHealthManager(bot))
    bot.tts_manager = bot._resolve_dependency("tts_manager", lambda: PackagedTTSManager(bot))
    bot.avatar_manager = bot._resolve_dependency("avatar_manager", lambda: PackagedAvatarManager(bot))
    bot.calendar_manager = bot._resolve_dependency("calendar_manager", lambda: PackagedCalendarManager(bot))
    bot.email_manager = bot._resolve_dependency("email_manager", lambda: PackagedEmailManager(bot))

    bot.model_runtime = bot._resolve_dependency("model_runtime", lambda: PackagedRuntimeModelManager(bot.bot_context))
    bot.runtime_client = bot._resolve_dependency("runtime_client", lambda: PackagedRuntimeClientManager(bot.bot_context))

    runtime_state_bundle = bot._resolve_runtime_state_bundle()
    bot._runtime_state_store = runtime_state_bundle.get("store")
    bot._runtime_event_bus = runtime_state_bundle.get("event_bus")
    bot._runtime_state = runtime_state_bundle["container"]
    bot.runtime_state_manager = bot._resolve_dependency(
        "runtime_state_manager",
        lambda: PackagedRuntimeStateManager(bot, bot._runtime_state),
    )
    bot.session_summary_manager = bot._resolve_dependency(
        "session_summary_manager",
        lambda: PackagedSessionSummaryManager(bot.bot_context),
    )


_EXTRA_MANAGER_ATTRS = ("internal_state_manager",)


def build_manager_registry(bot: Any) -> ServiceRegistry:
    """Return a ``ServiceRegistry`` populated with all wired bot manager instances.

    Walks ``DadBot._MANAGER_DELEGATE_CHAIN`` (plus a short list of extra manager
    attributes not in the chain) and registers each instance by name.  Uses
    ``object.__getattribute__`` so that the lookup never recurses through
    ``DadBot.__getattr__``.

    Call this after both ``wire_bootstrap_managers`` and ``wire_runtime_managers``
    have run so that every manager is present on the bot.
    """
    registry = ServiceRegistry()
    chain = getattr(type(bot), "_MANAGER_DELEGATE_CHAIN", ())
    for manager_name in (*chain, *_EXTRA_MANAGER_ATTRS):
        try:
            manager = object.__getattribute__(bot, manager_name)
        except AttributeError:
            continue
        if manager is not None:
            registry.register(manager_name, manager)
    return registry


def wire_bot_managers(bot: Any) -> None:
    """Compatibility helper for callers that want full manager wiring in one call."""
    wire_bootstrap_managers(bot)
    wire_runtime_managers(bot)


class DadBotServiceContainer:
    """Structured runtime container for DadBot managers and orchestration services.

    This wraps the existing wiring helpers so the facade can depend on a single
    container object without forcing a broad manager-construction rewrite.
    """

    def __init__(self, bot: Any):
        self.bot = bot
        self.registry = ServiceRegistry()
        self._turn_orchestrator = None

    def wire_bootstrap(self) -> None:
        wire_bootstrap_managers(self.bot)
        self.refresh_registry()

    def wire_runtime(self) -> None:
        wire_runtime_managers(self.bot)
        self.refresh_registry()

    def refresh_registry(self) -> ServiceRegistry:
        self.registry = build_manager_registry(self.bot)
        for name, instance in self.registry._services.items():
            setattr(self, name, instance)
        return self.registry

    def get_provider(self, name: str) -> Any | None:
        return self.registry.get_provider(name)

    def get(self, name: str) -> Any:
        return self.registry.get(name)

    @property
    def turn_orchestrator(self):
        if self._turn_orchestrator is None:
            from dadbot.core.orchestrator import DadBotOrchestrator

            self._turn_orchestrator = DadBotOrchestrator(
                config_path=self.bot.config.turn_graph_config_path,
                bot=self.bot,
                strict=bool(getattr(self.bot.config, "strict_graph_mode", True)),
            )
            self.bot._turn_orchestrator = self._turn_orchestrator
        return self._turn_orchestrator

    @property
    def service_registry(self) -> ServiceRegistry:
        return self.registry

    def validate_facade(self, *, smoke: bool = False) -> None:
        """Validate DadBot facade wiring using the extracted validator."""
        validate_dadbot_facade(self.bot, smoke=smoke)
