"""Protocol definitions for DadBot mixin contracts (Phase 1: Hardening)"""

from typing import Any, Protocol

from dadbot.contracts import AttachmentList


class TurnOrchestratorProvider(Protocol):
    """Protocol for objects that provide turn orchestration."""
    
    def _get_turn_orchestrator(self) -> Any:
        """Get the active turn orchestrator instance."""
        ...


class GraphFailureHandlerProvider(Protocol):
    """Protocol for objects that can handle graph failures."""
    
    active_thread_id: str
    tenant_id: str
    _turn_orchestrator: Any | None
    _strict_graph_mode: bool
    _current_turn_time_base: Any
    
    def _get_turn_orchestrator(self) -> Any:
        """Get the turn orchestrator."""
        ...
    
    def _append_signoff(self, message: str) -> str:
        """Append signature to a user-facing message."""
        ...


class ExecutionKernelProvider(Protocol):
    """Protocol for objects that provide execution kernel access."""
    
    def get_kernel(self) -> Any:
        """Get the active execution kernel."""
        ...


class PersistenceProvider(Protocol):
    """Protocol for objects that provide persistence services."""
    
    def get_persistence_service(self) -> Any:
        """Get the persistence service."""
        ...
