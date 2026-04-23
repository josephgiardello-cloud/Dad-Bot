from __future__ import annotations

import logging
from typing import Any

from dadbot.managers.conversation_persistence import ConversationPersistenceManager

logger = logging.getLogger(__name__)


class PersistenceService:
    """Service wrapper for durable turn/session persistence.

    The ``finalize_turn`` method is the atomic commit point for the SaveNode.
    It delegates to ``TurnService.finalize_user_turn``, which appends
    conversation history, schedules background maintenance, runs internal
    reflection, takes a health snapshot, and persists the session — all in a
    single call so no partial-state is ever written to disk.
    """

    def __init__(self, persistence_manager: ConversationPersistenceManager, turn_service: Any = None):
        self.persistence_manager = persistence_manager
        # Wired by ServiceRegistry.boot() after wire_runtime_managers has run.
        self.turn_service = turn_service

    def finalize_turn(self, turn_context: Any, result: Any) -> tuple:
        """Atomically commit history, maintenance, reflection, health snapshot, and persistence."""
        # Session exit was already handled inside prepare_user_turn_async — skip double-commit.
        if turn_context.state.get("already_finalized"):
            if isinstance(result, tuple) and len(result) >= 2:
                return result
            return (str(result or ""), bool(turn_context.state.get("should_end", False)))

        turn_text = turn_context.state.get("turn_text") or turn_context.user_input
        mood = turn_context.state.get("mood") or "neutral"
        norm_attachments = turn_context.state.get("norm_attachments") or turn_context.attachments
        reply = result[0] if isinstance(result, tuple) else str(result or "")

        if self.turn_service is not None:
            try:
                return self.turn_service.finalize_user_turn(turn_text, mood, reply, norm_attachments)
            except Exception as exc:
                logger.error("PersistenceService.finalize_turn: finalize_user_turn failed: %s", exc)

        # Fallback: basic persistence only
        try:
            self.persistence_manager.persist_conversation()
        except Exception as exc:
            logger.error("PersistenceService.finalize_turn: persist_conversation failed: %s", exc)
        return (reply, False)

    def save_turn(self, turn_context: Any, result: Any) -> None:
        snapshot_builder = getattr(turn_context, "snapshot", None)
        if callable(snapshot_builder):
            self.persistence_manager.persist_conversation_snapshot(snapshot_builder(result))
            return
        self.persistence_manager.persist_conversation()

    def save_graph_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        try:
            self.persistence_manager.persist_graph_checkpoint(checkpoint)
        except Exception as exc:
            logger.error("PersistenceService.save_graph_checkpoint failed: %s", exc)

    def save_turn_event(self, event: dict[str, Any]) -> None:
        try:
            self.persistence_manager.persist_turn_event(event)
        except Exception as exc:
            logger.error("PersistenceService.save_turn_event failed: %s", exc)

    def list_turn_events(self, trace_id: str, limit: int = 0) -> list[dict[str, Any]]:
        try:
            return self.persistence_manager.list_turn_events(trace_id=trace_id, limit=limit)
        except Exception as exc:
            logger.error("PersistenceService.list_turn_events failed: %s", exc)
            return []

    def replay_turn_events(self, trace_id: str) -> dict[str, Any]:
        try:
            return self.persistence_manager.replay_turn_events(trace_id=trace_id)
        except Exception as exc:
            logger.error("PersistenceService.replay_turn_events failed: %s", exc)
            return {"trace_id": str(trace_id or ""), "events": [], "replayed_state": {}}

    def validate_replay_determinism(self, trace_id: str, expected_lock_hash: str = "") -> dict[str, Any]:
        try:
            return self.persistence_manager.validate_replay_determinism(
                trace_id=trace_id,
                expected_lock_hash=expected_lock_hash,
            )
        except Exception as exc:
            logger.error("PersistenceService.validate_replay_determinism failed: %s", exc)
            return {
                "trace_id": str(trace_id or ""),
                "consistent": False,
                "observed_lock_hash": "",
                "expected_lock_hash": str(expected_lock_hash or ""),
                "matches_expected": False,
                "lock_hashes": [],
            }

    def persist_conversation(self) -> None:
        self.persistence_manager.persist_conversation()
