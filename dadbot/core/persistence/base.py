"""Abstract base classes for checkpoint persistence adapters."""

from abc import ABC, abstractmethod
from typing import Any


class CheckpointError(Exception):
    """Base exception for checkpoint operations."""


class CheckpointIntegrityError(CheckpointError):
    """Raised when checkpoint hash-chain or manifest verification fails."""


class CheckpointNotFoundError(CheckpointError):
    """Raised when a checkpoint cannot be loaded."""


class AbstractCheckpointer(ABC):
    """Abstract interface for checkpoint persistence backends.

    Concrete implementations handle:
    - Saving and loading checkpoints with hash-chain verification
    - Manifest drift detection on load
    - Pruning old checkpoints to prevent unbounded growth
    - Tenant-owned checkpoint access via (tenant_id, session_id, trace_id)

    PostgreSQL and Redis adapters are opt-in and require trusted-local mode.
    """

    @abstractmethod
    def save_checkpoint(
        self,
        session_id: str,
        trace_id: str,
        checkpoint: dict[str, Any],
        manifest: dict[str, Any],
        *,
        tenant_id: str | None = None,
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

    @abstractmethod
    def load_checkpoint(
        self,
        session_id: str,
        trace_id: str | None = None,
        *,
        current_manifest: dict[str, Any] | None = None,
        strict: bool = False,
        tenant_id: str | None = None,
    ) -> dict[str, Any]:
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

    @abstractmethod
    def prune_old_checkpoints(
        self,
        session_id: str,
        keep_count: int = 10,
        older_than_days: int | None = None,
        *,
        tenant_id: str | None = None,
    ) -> int:
        """Delete old checkpoints, keeping only the most recent N.

        Args:
            session_id: Session identifier
            keep_count: Number of checkpoints to retain (default 10)

        Returns:
            Number of checkpoints deleted

        """

    @abstractmethod
    def delete_session(self, session_id: str, *, tenant_id: str | None = None) -> int:
        """Delete all checkpoints for a session.

        Args:
            session_id: Session identifier

        Returns:
            Number of checkpoints deleted

        """

    @abstractmethod
    def migrate(self) -> None:
        """Apply schema migration(s) for the checkpoint backend."""

    # Compatibility aliases for simplified persistence naming.
    def save(
        self,
        session_id: str,
        trace_id: str,
        checkpoint: dict[str, Any],
        manifest: dict[str, Any],
        *,
        tenant_id: str | None = None,
    ) -> bool:
        return self.save_checkpoint(session_id, trace_id, checkpoint, manifest, tenant_id=tenant_id)

    def load(
        self,
        session_id: str,
        trace_id: str | None = None,
        *,
        current_manifest: dict[str, Any] | None = None,
        strict: bool = False,
        tenant_id: str | None = None,
    ) -> dict[str, Any]:
        return self.load_checkpoint(
            session_id,
            trace_id,
            current_manifest=current_manifest,
            strict=strict,
            tenant_id=tenant_id,
        )

    def checkpoint(
        self,
        session_id: str,
        trace_id: str | None = None,
        *,
        current_manifest: dict[str, Any] | None = None,
        strict: bool = False,
        tenant_id: str | None = None,
    ) -> dict[str, Any]:
        return self.load(
            session_id,
            trace_id,
            current_manifest=current_manifest,
            strict=strict,
            tenant_id=tenant_id,
        )


class AbstractAsyncCheckpointer(ABC):
    """Async counterpart for checkpoint persistence backends."""

    @abstractmethod
    async def save_checkpoint(
        self,
        session_id: str,
        trace_id: str,
        checkpoint: dict[str, Any],
        manifest: dict[str, Any],
        *,
        tenant_id: str | None = None,
    ) -> bool:
        ...

    @abstractmethod
    async def load_checkpoint(
        self,
        session_id: str,
        trace_id: str | None = None,
        *,
        current_manifest: dict[str, Any] | None = None,
        strict: bool = False,
        tenant_id: str | None = None,
    ) -> dict[str, Any]:
        ...

    @abstractmethod
    async def prune_old_checkpoints(
        self,
        session_id: str,
        keep_count: int = 10,
        older_than_days: int | None = None,
        *,
        tenant_id: str | None = None,
    ) -> int:
        ...

    @abstractmethod
    async def migrate(self) -> None:
        ...

    async def save(
        self,
        session_id: str,
        trace_id: str,
        checkpoint: dict[str, Any],
        manifest: dict[str, Any],
        *,
        tenant_id: str | None = None,
    ) -> bool:
        return await self.save_checkpoint(
            session_id,
            trace_id,
            checkpoint,
            manifest,
            tenant_id=tenant_id,
        )

    async def load(
        self,
        session_id: str,
        trace_id: str | None = None,
        *,
        current_manifest: dict[str, Any] | None = None,
        strict: bool = False,
        tenant_id: str | None = None,
    ) -> dict[str, Any]:
        return await self.load_checkpoint(
            session_id,
            trace_id,
            current_manifest=current_manifest,
            strict=strict,
            tenant_id=tenant_id,
        )

    async def checkpoint(
        self,
        session_id: str,
        trace_id: str | None = None,
        *,
        current_manifest: dict[str, Any] | None = None,
        strict: bool = False,
        tenant_id: str | None = None,
    ) -> dict[str, Any]:
        return await self.load(
            session_id,
            trace_id,
            current_manifest=current_manifest,
            strict=strict,
            tenant_id=tenant_id,
        )


def delete(session_id: str) -> int:
    """Module-level alias retained for canonical PersistenceBase contract checks."""
    _ = session_id
    raise RuntimeError("Persistence delete alias must be implemented by a concrete backend")
