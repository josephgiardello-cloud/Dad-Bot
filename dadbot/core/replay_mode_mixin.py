"""Replay mode detection and initialization for checkpoint-driven deterministic replay.

Provides infrastructure for:
- Detecting if a turn should execute in replay mode (checkpoint exists)
- Loading checkpoint data into turn context
- Restoring tool_io_ledger for fast lookups
"""

from __future__ import annotations

import logging
from typing import Any

from dadbot.core.persistence_record_schema import PersistenceSchemaError
from dadbot.core.runtime_errors import NON_FATAL_RUNTIME_EXCEPTIONS
from dadbot.core.tool_recording import ToolIOLedger

logger = logging.getLogger(__name__)


class ReplayModeMixin:
    """Mixin for replay mode detection and checkpoint restoration."""

    def _detect_and_prepare_replay_mode(
        self,
        session_id: str,
        trace_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Detect if turn should run in replay mode and prepare metadata.

        Returns:
            Updated metadata dict with replay_mode flag and restored checkpoint data.
        """
        if metadata is None:
            metadata = {}

        # Try to load checkpoint
        checkpoint = self._try_load_checkpoint(session_id=session_id, trace_id=trace_id)

        if checkpoint is not None:
            # Replay mode: checkpoint exists
            metadata["replay_mode"] = True
            metadata["checkpoint_available"] = True

            # Restore tool_io_ledger from checkpoint for fast lookup during execute_tool()
            if "tool_io_ledger" in checkpoint:
                metadata["_tool_io_ledger"] = ToolIOLedger.from_checkpoint(checkpoint["tool_io_ledger"])
                metadata["tool_io_ledger_restored"] = True

            logger.debug(
                "Replay mode enabled for session %s: checkpoint found",
                session_id,
            )
        else:
            # Live mode: no checkpoint
            metadata["replay_mode"] = False
            metadata["checkpoint_available"] = False
            logger.debug(
                "Live mode: no checkpoint found for session %s",
                session_id,
            )

        return metadata

    def _try_load_checkpoint(
        self,
        session_id: str,
        trace_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Attempt to load checkpoint from persistence layer.

        Returns:
            Checkpoint dict if found, None otherwise.
        """
        persistence_service = getattr(self, "services", None)
        get_persistence = getattr(persistence_service, "get_persistence_service", None)
        if not callable(get_persistence):
            return None
        try:
            persistence = get_persistence()
        except (AttributeError, RuntimeError) as exc:
            logger.debug("Could not resolve checkpoint persistence: %s", exc)
            return None
        load_latest = getattr(persistence, "load_latest_graph_checkpoint", None)
        if not callable(load_latest):
            return None
        try:
            checkpoint = load_latest(
                trace_token=str(trace_id or ""),
                session_id=str(session_id or "").strip() or "default",
            )
        except PersistenceSchemaError:
            raise
        except NON_FATAL_RUNTIME_EXCEPTIONS as exc:
            logger.debug("Could not load checkpoint: %s", exc)
            return None
        return checkpoint if isinstance(checkpoint, dict) else None
