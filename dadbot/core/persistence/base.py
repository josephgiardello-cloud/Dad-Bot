"""Abstract base classes for checkpoint persistence adapters."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class CheckpointError(Exception):
    """Base exception for checkpoint operations."""
    pass


class CheckpointIntegrityError(CheckpointError):
    """Raised when checkpoint hash-chain or manifest verification fails."""
    pass


class CheckpointNotFoundError(CheckpointError):
    """Raised when a checkpoint cannot be loaded."""
    pass


class AbstractCheckpointer(ABC):
    """Abstract interface for checkpoint persistence backends.
    
    Implementations (SQLiteCheckpointer, PostgresCheckpointer) handle:
    - Saving and loading checkpoints with hash-chain verification
    - Manifest drift detection on load
    - Pruning old checkpoints to prevent unbounded growth
    - Concurrent access via session_id + trace_id composite key
    """

    @abstractmethod
    def save_checkpoint(
        self,
        session_id: str,
        trace_id: str,
        checkpoint: Dict[str, Any],
        manifest: Dict[str, Any],
    ) -> bool:
        """Save a checkpoint with integrity verification.
        
        Args:
            session_id: Session identifier (user_id or conversation thread)
            trace_id: Unique turn identifier (prevents duplicate replay)
            checkpoint: Checkpoint dict with keys:
                - checkpoint_hash: sha256 of (state + prev_checkpoint_hash)
                - prev_checkpoint_hash: sha256 of previous checkpoint (or empty string if first)
                - status: turn stage when saved (e.g., "completed")
                - error: error message if present, else null
                - Full state blob (serialized)
            manifest: Determinism manifest with keys:
                - python_version
                - env_hash
                - dependency_versions
                - timezone
        
        Returns:
            True if save was successful
            
        Raises:
            CheckpointError: if save fails (DB error, permissions, etc.)
        """
        pass

    @abstractmethod
    def load_checkpoint(
        self,
        session_id: str,
        trace_id: Optional[str] = None,
        *,
        current_manifest: Optional[Dict[str, Any]] = None,
        strict: bool = False,
    ) -> Dict[str, Any]:
        """Load and verify a checkpoint.
        
        Args:
            session_id: Session identifier
            trace_id: Optional specific trace to load; if None, loads most recent
        
        Returns:
            Checkpoint dict (same structure as saved)
            
        Raises:
            CheckpointNotFoundError: if checkpoint does not exist
            CheckpointIntegrityError: if hash-chain or manifest verification fails
        """
        pass

    @abstractmethod
    def prune_old_checkpoints(
        self,
        session_id: str,
        keep_count: int = 10,
        older_than_days: int | None = None,
    ) -> int:
        """Delete old checkpoints, keeping only the most recent N.
        
        Args:
            session_id: Session identifier
            keep_count: Number of checkpoints to retain (default 10)
        
        Returns:
            Number of checkpoints deleted
        """
        pass

    @abstractmethod
    def delete_session(self, session_id: str) -> int:
        """Delete all checkpoints for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Number of checkpoints deleted
        """
        pass

    @abstractmethod
    def migrate(self) -> None:
        """Apply schema migration(s) for the checkpoint backend."""
        pass


class AbstractAsyncCheckpointer(ABC):
    """Async counterpart for checkpoint persistence backends."""

    @abstractmethod
    async def save_checkpoint(
        self,
        session_id: str,
        trace_id: str,
        checkpoint: Dict[str, Any],
        manifest: Dict[str, Any],
    ) -> bool:
        pass

    @abstractmethod
    async def load_checkpoint(
        self,
        session_id: str,
        trace_id: Optional[str] = None,
        *,
        current_manifest: Optional[Dict[str, Any]] = None,
        strict: bool = False,
    ) -> Dict[str, Any]:
        pass

    @abstractmethod
    async def prune_old_checkpoints(
        self,
        session_id: str,
        keep_count: int = 10,
        older_than_days: int | None = None,
    ) -> int:
        pass

    @abstractmethod
    async def migrate(self) -> None:
        pass
