from __future__ import annotations

import logging
import time
from typing import Any

from dadbot.core.graph import FatalTurnError, MemoryMutationOp, MutationIntent, MutationKind
from dadbot.managers.conversation_persistence import ConversationPersistenceManager

logger = logging.getLogger(__name__)


class PersistenceService:
    """Service wrapper for durable turn/session persistence.

    The ``finalize_turn`` method is the atomic commit point for the SaveNode.
    It delegates to ``TurnService.finalize_user_turn``, which appends
    conversation history, schedules background maintenance, runs internal
    reflection, takes a health snapshot, and persists the session â€” all in a
    single call so no partial-state is ever written to disk.
    """

    def __init__(self, persistence_manager: ConversationPersistenceManager, turn_service: Any = None):
        self.persistence_manager = persistence_manager
        # Wired by ServiceRegistry.boot() after wire_runtime_managers has run.
        self.turn_service = turn_service

    @staticmethod
    def _call_nonfatal(callable_obj: Any, *args: Any, **kwargs: Any) -> Any:
        if not callable(callable_obj):
            return None
        try:
            return callable_obj(*args, **kwargs)
        except Exception as exc:
            logger.warning("PersistenceService post-finalize hook failed (non-fatal): %s", exc)
            return None

    def _apply_memory_decay(self, memory_manager: Any, turn_context: Any) -> None:
        """Apply deterministic memory decay after consolidation, before graph sync.

        Uses MemoryDecayPolicy â€” no datetime.now(), no external clocks.
        Non-fatal: any failure is logged and silently skipped.
        """
        from dadbot.memory.decay_policy import DecayResult, MemoryDecayPolicy

        try:
            entries = list(memory_manager.consolidated_memories())
            if not entries:
                return
            policy = MemoryDecayPolicy()
            result: DecayResult = policy.apply(entries, turn_context)
            if not result.pruned and not result.weakened:
                return

            pruned_ids = set(result.pruned)
            weakened_ids = set(result.weakened)
            updated: list[Any] = []
            for entry in entries:
                eid = str(entry.get("id", ""))
                if eid in pruned_ids:
                    continue
                if eid in weakened_ids:
                    entry = dict(entry)
                    old = float(entry.get("importance_score", 0.0) or 0.0)
                    entry["importance_score"] = round(max(0.0, old * policy.weaken_factor), 4)
                updated.append(entry)
            memory_manager.mutate_memory_store(consolidated_memories=updated)

            # Surface result to graph state for observability / replay auditing
            state = getattr(turn_context, "state", None)
            if isinstance(state, dict):
                state["memory_decay_result"] = {
                    "pruned": result.pruned,
                    "weakened": result.weakened,
                    "unchanged_count": len(result.unchanged),
                    "total_score_map": result.total_score_map,
                }
        except Exception as exc:
            logger.warning("PersistenceService._apply_memory_decay failed (non-fatal): %s", exc)

    def _apply_pending_save_boundary_mutations(self, runtime: Any, turn_context: Any) -> None:
        state = getattr(turn_context, "state", None)
        if not isinstance(state, dict):
            return

        pending_moods = list(state.get("_pending_mood_updates") or [])
        pending_relationship = list(state.get("_pending_relationship_updates") or [])
        if not pending_moods and not pending_relationship:
            # Backward compatibility with older deferred key.
            legacy = list(state.get("_deferred_turn_state_updates") or [])
            if legacy:
                pending_moods = [{"mood": str(item.get("mood") or "neutral")} for item in legacy]
                pending_relationship = [
                    {
                        "op": "update",
                        "user_input": str(item.get("user_input") or ""),
                        "mood": str(item.get("mood") or "neutral"),
                    }
                    for item in legacy
                    if str(item.get("user_input") or "").strip()
                ]

        memory = getattr(runtime, "memory", None)
        if memory is None:
            raise RuntimeError("SaveNode strict mode requires runtime.memory")

        for item in pending_moods:
            mood = str(item.get("mood") or "neutral")
            memory.save_mood_state(mood)

        state["_pending_mood_updates"] = []
        state["_pending_relationship_updates"] = []
        state["_deferred_turn_state_updates"] = []

    @staticmethod
    def _flush_background_memory_store_patch_queue(runtime: Any) -> int:
        queue = getattr(runtime, "_background_memory_store_patch_queue", None)
        if not isinstance(queue, list) or not queue:
            return 0
        pending = list(queue)
        queue.clear()
        applied = 0
        for patch in pending:
            if not isinstance(patch, dict):
                continue
            runtime.mutate_memory_store(**patch)
            applied += 1
        return applied

    def _commit_post_finalize_side_effects(self, turn_context: Any) -> None:
        """Run strict SaveNode mutation sequence before final ledger commit."""
        service = self.turn_service
        runtime = None if service is None else getattr(service, "bot", None)
        if runtime is None:
            raise RuntimeError("SaveNode strict mode requires an attached turn_service runtime")

        memory_coordinator = getattr(runtime, "memory_coordinator", None)
        if memory_coordinator is None:
            raise RuntimeError("SaveNode strict mode requires memory_coordinator")

        # --- Drain MutationQueue FIRST at the canonical SaveNode commit boundary ---
        # Every mutation queued outside this boundary (deferred from earlier stages or
        # direct-path callers) must execute here. Any failure is a hard fail — nothing
        # is silently dropped.
        mutation_queue = getattr(turn_context, "mutation_queue", None)
        if mutation_queue is not None:
            def _dispatch_mutation_intent(intent: Any) -> None:
                if not isinstance(intent, MutationIntent):
                    raise RuntimeError(f"MutationQueue received non-MutationIntent payload: {type(intent).__name__}")
                intent_type = intent.type
                payload = dict(intent.payload or {})
                source = str(intent.source or "")
                if intent_type is MutationKind.MEMORY:
                    op = str(payload.get("op") or "").strip().lower()
                    if op != MemoryMutationOp.SAVE_MOOD_STATE.value:
                        raise RuntimeError(
                            f"MutationIntent(type=memory, source={source!r}): unsupported op={op!r}"
                        )
                    mood = str(payload.get("mood") or "neutral")
                    memory = getattr(runtime, "memory", None)
                    if memory is None:
                        raise RuntimeError(
                            f"MutationIntent(type=memory, source={source!r}): runtime.memory unavailable"
                        )
                    memory.save_mood_state(mood)
                elif intent_type is MutationKind.RELATIONSHIP:
                    raise RuntimeError(
                        "MutationIntent(type=relationship) rejected: relationship subsystem is projection-only"
                    )
                elif intent_type is MutationKind.GRAPH:
                    memory_manager = getattr(runtime, "memory_manager", None)
                    graph_manager = getattr(memory_manager, "graph_manager", None) if memory_manager else None
                    if graph_manager is None:
                        raise RuntimeError(f"MutationIntent(type=graph, source={source!r}): graph_manager unavailable")
                    _fn = getattr(graph_manager, "apply_mutation", None)
                    if callable(_fn):
                        _fn(payload, turn_context=turn_context)
                    else:
                        raise RuntimeError(f"MutationIntent(type=graph, source={source!r}): graph_manager.apply_mutation not callable")
                else:
                    raise RuntimeError(f"MutationIntent: unknown type={intent_type!r} source={source!r}")

            mutation_queue.drain(_dispatch_mutation_intent, hard_fail_on_error=True)
            if not mutation_queue.is_empty():
                pending = mutation_queue.size()
                raise FatalTurnError(
                    "Mutation queue not fully drained"
                    f" (pending={pending}, trace_id={getattr(turn_context, 'trace_id', '')!r})"
                )
        # --------------------------------------------------------------------------

        # Apply any pending non-SaveNode mutation intents at the canonical commit boundary.
        self._apply_pending_save_boundary_mutations(runtime, turn_context)
        self._flush_background_memory_store_patch_queue(runtime)
        flush_deferred = getattr(self.persistence_manager, "flush_deferred_save_boundary_mutations", None)
        if callable(flush_deferred):
            self._call_nonfatal(flush_deferred, turn_context)

        consolidate_memories = getattr(memory_coordinator, "consolidate_memories", None)
        if not callable(consolidate_memories):
            raise RuntimeError("SaveNode strict mode requires memory_coordinator.consolidate_memories")
        memory_ops_started = time.perf_counter()
        consolidate_memories(turn_context=turn_context)

        apply_controlled_forgetting = getattr(memory_coordinator, "apply_controlled_forgetting", None)
        if not callable(apply_controlled_forgetting):
            raise RuntimeError("SaveNode strict mode requires memory_coordinator.apply_controlled_forgetting")
        apply_controlled_forgetting(turn_context=turn_context)
        memory_ops_ms = round((time.perf_counter() - memory_ops_started) * 1000, 3)
        if isinstance(getattr(turn_context, "state", None), dict):
            turn_context.state["_timing_memory_ops_ms"] = memory_ops_ms

        relationship_manager = getattr(runtime, "relationship_manager", None)
        materialize_projection = getattr(relationship_manager, "materialize_projection", None)
        if not callable(materialize_projection):
            raise RuntimeError("SaveNode strict mode requires relationship_projector.materialize_projection")
        materialize_projection(turn_context=turn_context)

        memory_manager = getattr(runtime, "memory_manager", None)
        graph_manager = getattr(memory_manager, "graph_manager", None) if memory_manager is not None else None
        sync_graph_store = getattr(graph_manager, "sync_graph_store", None)
        if not callable(sync_graph_store):
            raise RuntimeError("SaveNode strict mode requires memory_graph_manager.sync_graph_store")
        graph_sync_started = time.perf_counter()
        sync_graph_store(turn_context=turn_context)
        graph_sync_ms = round((time.perf_counter() - graph_sync_started) * 1000, 3)
        if isinstance(getattr(turn_context, "state", None), dict):
            turn_context.state["_timing_graph_sync_ms"] = graph_sync_ms

    def begin_transaction(self, turn_context: Any) -> None:
        temporal = getattr(turn_context, "temporal", None)
        if temporal is None:
            raise RuntimeError("TemporalNode required — execution invalid")
        state = getattr(turn_context, "state", None)
        if isinstance(state, dict):
            state["_save_transaction_active"] = True

    def apply_mutations(self, turn_context: Any) -> None:
        temporal = getattr(turn_context, "temporal", None)
        if temporal is None:
            raise RuntimeError("TemporalNode required — execution invalid")
        service = self.turn_service
        runtime = None if service is None else getattr(service, "bot", None)
        if runtime is None:
            raise RuntimeError("SaveNode strict mode requires turn_service.bot")
        previous_commit_active = bool(getattr(runtime, "_graph_commit_active", False))
        try:
            runtime._graph_commit_active = True
            self._commit_post_finalize_side_effects(turn_context)
        finally:
            runtime._graph_commit_active = previous_commit_active
        state = getattr(turn_context, "state", None)
        if isinstance(state, dict):
            state["_save_mutations_applied"] = True

    def commit_transaction(self, turn_context: Any) -> None:
        state = getattr(turn_context, "state", None)
        if isinstance(state, dict):
            state["_save_transaction_active"] = False
            state["_save_mutations_applied"] = False

    def rollback_transaction(self, turn_context: Any) -> None:
        state = getattr(turn_context, "state", None)
        if isinstance(state, dict):
            state["_save_transaction_active"] = False
            state["_save_mutations_applied"] = False

    def final_ledger_commit(
        self,
        turn_text: str,
        mood: str,
        reply: str,
        norm_attachments: Any,
    ) -> tuple:
        if self.turn_service is None:
            raise RuntimeError("SaveNode strict mode requires graph turn_service wiring in Phase 4")
        return self.turn_service.finalize_user_turn(turn_text, mood, reply, norm_attachments)

    def finalize_turn(self, turn_context: Any, result: Any) -> tuple:
        """Atomically commit history, maintenance, reflection, health snapshot, and persistence."""
        # Session exit was already handled inside prepare_user_turn_async â€” skip double-commit.
        if turn_context.state.get("already_finalized"):
            if isinstance(result, tuple) and len(result) >= 2:
                return result
            return (str(result or ""), bool(turn_context.state.get("should_end", False)))

        turn_text = turn_context.state.get("turn_text") or turn_context.user_input
        mood = turn_context.state.get("mood") or "neutral"
        norm_attachments = turn_context.state.get("norm_attachments") or turn_context.attachments
        reply = result[0] if isinstance(result, tuple) else str(result or "")

        service = self.turn_service
        if service is None:
            raise RuntimeError("Strict mode requires graph turn_service wiring in Phase 4")

        runtime = getattr(service, "bot", None)
        if runtime is None:
            raise RuntimeError("SaveNode strict mode requires turn_service.bot")
        if getattr(turn_context, "temporal", None) is None:
            raise RuntimeError("TemporalNode required — execution invalid")

        previous_temporal = getattr(runtime, "_current_turn_time_base", None)
        previous_commit_active = bool(getattr(runtime, "_graph_commit_active", False))
        finalize_started = time.perf_counter()
        try:
            runtime._current_turn_time_base = getattr(turn_context, "temporal", None)
            runtime._graph_commit_active = True

            # Strict sequence before final ledger commit.
            state = getattr(turn_context, "state", None)
            mutations_applied = bool(state.get("_save_mutations_applied")) if isinstance(state, dict) else False
            if not mutations_applied:
                self._commit_post_finalize_side_effects(turn_context)
            finalized = self.final_ledger_commit(turn_text, mood, reply, norm_attachments)
            return finalized
        except Exception as exc:
            logger.error("PersistenceService.finalize_turn strict-mode failure: %s", exc)
            raise
        finally:
            if isinstance(getattr(turn_context, "state", None), dict):
                turn_context.state["_timing_finalize_ms"] = round((time.perf_counter() - finalize_started) * 1000, 3)
            runtime._graph_commit_active = previous_commit_active
            runtime._current_turn_time_base = previous_temporal

    def save_turn(self, turn_context: Any, result: Any) -> None:
        snapshot_builder = getattr(turn_context, "snapshot", None)
        if callable(snapshot_builder):
            self.persistence_manager.persist_conversation_snapshot(snapshot_builder(result), turn_context=turn_context)
            return
        self.persistence_manager.persist_conversation()

    def save_graph_checkpoint(self, checkpoint: dict[str, Any], _skip_turn_event: bool = False) -> None:
        try:
            self.persistence_manager.persist_graph_checkpoint(checkpoint, _skip_turn_event=_skip_turn_event)
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
