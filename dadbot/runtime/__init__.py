"""Runtime decomposition modules."""

from .agent_driver_loop import AgentDriverLoop, DriverLoopPolicy, DriverLoopResult
from .context_pruner import ContextWindowPruner, PrunedContext, build_pruned_observation_hook
from .loop_snapshot import LoopSessionSnapshot, LoopSnapshotManager
from .semantic_memory_bridge import (
    MemoryConsolidationJob,
    MemoryIndexer,
    SemanticRetrievalHook,
    build_semantic_snippet_provider,
)
from .structured_output import (
    AgentPlan,
    SchemaValidationError,
    ToolCall,
    build_llm_reflection_hook,
    build_reflection_prompt,
    parse_agent_plan,
)

__all__ = [
    # Core loop
    "AgentDriverLoop",
    "DriverLoopPolicy",
    "DriverLoopResult",
    # Structured output / schema validation
    "AgentPlan",
    "ToolCall",
    "SchemaValidationError",
    "parse_agent_plan",
    "build_reflection_prompt",
    "build_llm_reflection_hook",
    # Context pruner
    "ContextWindowPruner",
    "PrunedContext",
    "build_pruned_observation_hook",
    # Semantic memory bridge
    "MemoryIndexer",
    "SemanticRetrievalHook",
    "build_semantic_snippet_provider",
    "MemoryConsolidationJob",
    # Session snapshot / resume
    "LoopSessionSnapshot",
    "LoopSnapshotManager",
]
