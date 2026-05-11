from __future__ import annotations

import concurrent.futures
import contextlib
import hashlib
import json
import logging
import os
import time
import uuid
from copy import deepcopy
from typing import Any

from dadbot.core.execution_context import (
    RuntimeTraceViolation,
    ensure_execution_trace_root,
    require_bound_core_state_for_mutation,
)
from dadbot.core.execution_ledger import IntegrityBreachError
from dadbot.core.kernel_mutation_gate import apply_event, emit_event
from dadbot.core.persistence import AbstractCheckpointer
from dadbot.core.post_commit_events import PostCommitEvent
from dadbot.core.runtime_errors import (
    NON_FATAL_RUNTIME_EXCEPTIONS,
    ExecutionStageError,
    InvariantViolation,
    PersistenceFailure,
)
from dadbot.managers.conversation_persistence import ConversationPersistenceManager
from dadbot.services._persistence_mixins import (
    POLICY_TRACE_EVENT_TYPE,
    StateDivergenceError,
    _AuthorityStateMixin,
    _BehavioralLedgerMixin,
    _IntegrityVerifyMixin,
    _LedgerOpsMixin,
    _MutationDispatchMixin,
)

logger = logging.getLogger(__name__)

# Re-export for backward compatibility: external code imports StateDivergenceError
# from this module.
__all__ = ["PersistenceService", "StateDivergenceError"]


class PersistenceService(
    _LedgerOpsMixin,
    _MutationDispatchMixin,
    _BehavioralLedgerMixin,
    _AuthorityStateMixin,
    _IntegrityVerifyMixin,
):
    """Service wrapper for durable turn/session persistence.

    The ``finalize_turn`` method is the atomic commit point for the SaveNode.
    It delegates to ``TurnService.finalize_user_turn``, which appends
    conversation history, schedules background maintenance, runs internal
    reflection, takes a health snapshot, and persists the session -- all in a
    single call so no partial-state is ever written to disk.
    """

    def __init__(
        self,
        persistence_manager: ConversationPersistenceManager,
        turn_service: Any = None,
    ):
        self.persistence_manager = persistence_manager
        # Wired by ServiceRegistry.boot() after wire_runtime_managers has run.
        self.turn_service = turn_service
        self.checkpointer: AbstractCheckpointer | None = None
        self.strict_mode: bool = False
        self._merkle_session_leaves: dict[str, list[str]] = {}
        self._async_checkpoint_pipeline_enabled: bool = (
            str(os.environ.get("DADBOT_ASYNC_CHECKPOINT_PIPELINE", "")).strip().lower()
            in {"1", "true", "on", "yes"}
        )
        self._checkpoint_executor: concurrent.futures.ThreadPoolExecutor | None = None
        self._checkpoint_futures: list[tuple[int, concurrent.futures.Future[Any]]] = []
        self._checkpoint_enqueue_sequence: int = 0
        self._checkpoint_drain_sequence: int = 0

    def _ensure_checkpoint_executor(self) -> concurrent.futures.ThreadPoolExecutor:
        if self._checkpoint_executor is None:
            self._checkpoint_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="dadbot-checkpoint",
            )
        return self._checkpoint_executor

    def _drain_async_checkpoint_queue(self, *, strict_error: bool) -> None:
        if not self._checkpoint_futures:
            return
        pending = list(self._checkpoint_futures)
        self._checkpoint_futures.clear()
        expected = int(self._checkpoint_drain_sequence) + 1
        for sequence_id, future in pending:
            if int(sequence_id) != int(expected):
                raise InvariantViolation(
                    "Async checkpoint ordering contract violated",
                    context={
                        "expected_sequence": int(expected),
                        "actual_sequence": int(sequence_id),
                    },
                )
            try:
                future.result()
                self._checkpoint_drain_sequence = int(sequence_id)
                expected = int(sequence_id) + 1
            except NON_FATAL_RUNTIME_EXCEPTIONS as exc:
                logger.error("PersistenceService async checkpoint save failed: %s", exc)
                if strict_error:
                    raise PersistenceFailure(
                        "PersistenceService async checkpoint save failed",
                        context={"error": str(exc), "sequence_id": int(sequence_id)},
                    ) from exc

    def _enqueue_async_checkpoint(
        self,
        *,
        session_id: str,
        trace_id: str,
        checkpoint: dict[str, Any],
        manifest: dict[str, Any],
    ) -> None:
        if self.checkpointer is None:
            return
        self._checkpoint_enqueue_sequence = int(self._checkpoint_enqueue_sequence) + 1
        sequence_id = int(self._checkpoint_enqueue_sequence)
        if sequence_id <= int(self._checkpoint_drain_sequence):
            raise InvariantViolation(
                "Async checkpoint enqueue sequence must advance monotonically",
                context={
                    "enqueue_sequence": sequence_id,
                    "drain_sequence": int(self._checkpoint_drain_sequence),
                },
            )
        executor = self._ensure_checkpoint_executor()
        future = executor.submit(
            self.checkpointer.save_checkpoint,
            session_id=session_id,
            trace_id=trace_id,
            checkpoint=dict(checkpoint),
            manifest=dict(manifest),
        )
        self._checkpoint_futures.append((sequence_id, future))

    @staticmethod
    def _stable_hash(payload: Any) -> str:
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str).encode(
                "utf-8",
            ),
        ).hexdigest()

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self._drain_async_checkpoint_queue(strict_error=False)
        if self._checkpoint_executor is not None:
            with contextlib.suppress(Exception):
                self._checkpoint_executor.shutdown(wait=False, cancel_futures=False)

    @staticmethod
    def _json_safe(payload: Any) -> Any:
        return json.loads(
            json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str),
        )

    def set_checkpointer(self, checkpointer: AbstractCheckpointer | None) -> None:
        self._drain_async_checkpoint_queue(strict_error=False)
        if self._checkpoint_executor is not None:
            with contextlib.suppress(Exception):
                self._checkpoint_executor.shutdown(wait=True, cancel_futures=False)
            self._checkpoint_executor = None
        self.checkpointer = checkpointer

    @staticmethod
    def _call_nonfatal(callable_obj: Any, *args: Any, **kwargs: Any) -> Any:
        if not callable(callable_obj):
            return None
        try:
            return callable_obj(*args, **kwargs)
        except NON_FATAL_RUNTIME_EXCEPTIONS as exc:
            logger.warning(
                "PersistenceService post-finalize hook failed (non-fatal): %s",
                exc,
            )
            return None

    def _apply_memory_decay(self, memory_manager: Any, ctx: Any) -> Any:
        """Persistence boundary guard.

        Memory decay and lifecycle evolution are post-commit responsibilities
        and must run in post-commit worker / maintenance services only.
        """
        raise ExecutionStageError(
            "PersistenceService._apply_memory_decay is not allowed",
            context={"required_path": "post-commit worker or maintenance service"},
        )

    def _apply_pending_save_boundary_mutations(
        self,
        runtime: Any,
        turn_context: Any,
    ) -> None:
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
            raise PersistenceFailure("SaveNode strict mode requires runtime.memory")

        for item in pending_moods:
            mood = str(item.get("mood") or "neutral")
            memory.save_mood_state(mood)

        clear_event = emit_event(
            "MUTATION_EVENT",
            {
                "op": "dict_update",
                "updates": {
                    "_pending_mood_updates": [],
                    "_pending_relationship_updates": [],
                    "_deferred_turn_state_updates": [],
                },
            },
            source="PersistenceService._apply_pending_save_boundary_mutations",
        )
        turn_context.state = apply_event(
            clear_event,
            state,
            lambda current, evt: {**current, **dict(evt.payload.get("updates") or {})},
        )

    def _publish_post_commit_ready(self, runtime: Any, turn_context: Any) -> None:
        """Publish the post-commit readiness event and return immediately."""
        event_bus = getattr(runtime, "_runtime_event_bus", None)
        publish = getattr(event_bus, "publish", None)
        if not callable(publish):
            logger.warning(
                "Post-commit worker unavailable: runtime event bus missing publish(); skipping post-commit event"
            )
            return

        metadata = getattr(turn_context, "metadata", None)
        control_plane = dict(metadata.get("control_plane") or {}) if isinstance(metadata, dict) else {}
        publish(
            PostCommitEvent(
                session_id=str(control_plane.get("session_id") or "default"),
                trace_id=str(getattr(turn_context, "trace_id", "") or ""),
                tenant_id=str(
                    getattr(getattr(runtime, "config", None), "tenant_id", "default")
                    or "default"
                ),
                payload={"turn_context": turn_context},
            )
        )

    def _commit_post_finalize_side_effects(self, turn_context: Any) -> None:
        """Run strict SaveNode mutation sequence before final ledger commit."""
        service = self.turn_service
        runtime = None if service is None else getattr(service, "bot", None)
        if runtime is None:
            raise PersistenceFailure(
                "SaveNode strict mode requires an attached turn_service runtime",
            )

        # --- Drain MutationQueue FIRST at the canonical SaveNode commit boundary ---
        # Every mutation queued outside this boundary (deferred from earlier stages or
        # direct-path callers) must execute here. Any failure is a hard fail -- nothing
        # is silently dropped.
        self._drain_mutation_queue(runtime, turn_context)
        # --------------------------------------------------------------------------

        # Apply any pending non-SaveNode mutation intents at the canonical commit boundary.
        self._apply_pending_save_boundary_mutations(runtime, turn_context)
        self._flush_background_memory_store_patch_queue(runtime)
        flush_deferred = getattr(
            self.persistence_manager,
            "flush_deferred_save_boundary_mutations",
            None,
        )
        if callable(flush_deferred):
            self._call_nonfatal(flush_deferred, turn_context)

        relationship_manager = getattr(runtime, "relationship_manager", None)
        materialize_projection = getattr(
            relationship_manager,
            "materialize_projection",
            None,
        )
        if not callable(materialize_projection):
            raise PersistenceFailure(
                "SaveNode strict mode requires relationship_projector.materialize_projection",
            )
        materialize_projection(turn_context=turn_context)

        memory_manager = getattr(runtime, "memory_manager", None)
        graph_manager = getattr(memory_manager, "graph_manager", None) if memory_manager is not None else None
        sync_graph_store = getattr(graph_manager, "sync_graph_store", None)
        if not callable(sync_graph_store):
            raise PersistenceFailure(
                "SaveNode strict mode requires memory_graph_manager.sync_graph_store",
            )
        graph_sync_started = time.perf_counter()
        sync_graph_store(turn_context=turn_context)
        graph_sync_ms = round((time.perf_counter() - graph_sync_started) * 1000, 3)
        if isinstance(getattr(turn_context, "state", None), dict):
            turn_context.state["_timing_graph_sync_ms"] = graph_sync_ms

        # Post-commit publish is intentionally performed by finalize_turn after
        # checkpoint/checkpointer writes complete to avoid worker-thread state
        # mutations racing with in-flight finalize snapshotting.

    @staticmethod
    def _capture_transaction_snapshot(
        runtime: Any,
        turn_context: Any,
    ) -> dict[str, Any]:
        graph_manager = getattr(
            getattr(runtime, "memory_manager", None),
            "graph_manager",
            None,
        )
        graph_snapshot = {"nodes": [], "edges": [], "updated_at": None}
        background_patch_queue = getattr(
            runtime,
            "_background_memory_store_patch_queue",
            None,
        )
        if graph_manager is not None:
            snapshot_builder = getattr(graph_manager, "graph_snapshot", None)
            if callable(snapshot_builder):
                graph_snapshot = snapshot_builder() or graph_snapshot

        return {
            "memory_store": deepcopy(dict(getattr(runtime, "MEMORY_STORE", {}) or {})),
            "graph_snapshot": deepcopy(graph_snapshot),
            "session_state": deepcopy(runtime.snapshot_session_state()),
            "last_turn_pipeline": deepcopy(
                dict(getattr(runtime, "_last_turn_pipeline", {}) or {}),
            ),
            "background_patch_queue": deepcopy(list(background_patch_queue))
            if isinstance(background_patch_queue, list)
            else None,
            "turn_state": deepcopy(dict(getattr(turn_context, "state", {}) or {})),
            "metadata": deepcopy(dict(getattr(turn_context, "metadata", {}) or {})),
        }

    @staticmethod
    def _restore_transaction_snapshot(
        runtime: Any,
        turn_context: Any,
        snapshot: dict[str, Any],
    ) -> None:
        require_bound_core_state_for_mutation(
            source="PersistenceService._restore_transaction_snapshot",
            changed_keys=["memory_store", "graph_snapshot", "session_state"],
        )
        session_state = dict(snapshot.get("session_state", {}) or {})
        if session_state:
            runtime.load_session_state_snapshot(deepcopy(session_state))
        else:
            runtime.MEMORY_STORE = deepcopy(
                dict(snapshot.get("memory_store", {}) or {}),
            )
        runtime._last_turn_pipeline = deepcopy(
            dict(snapshot.get("last_turn_pipeline", {}) or {}),
        )
        background_patch_queue = snapshot.get("background_patch_queue")
        if isinstance(background_patch_queue, list):
            runtime._background_memory_store_patch_queue = deepcopy(
                background_patch_queue,
            )

        state = getattr(turn_context, "state", None)
        if isinstance(state, dict):
            state.clear()
            state.update(deepcopy(dict(snapshot.get("turn_state", {}) or {})))

        metadata = getattr(turn_context, "metadata", None)
        if isinstance(metadata, dict):
            metadata.clear()
            metadata.update(deepcopy(dict(snapshot.get("metadata", {}) or {})))

        graph_snapshot = dict(snapshot.get("graph_snapshot", {}) or {})
        graph_manager = getattr(
            getattr(runtime, "memory_manager", None),
            "graph_manager",
            None,
        )
        backend = getattr(graph_manager, "_graph_store_backend", None) if graph_manager is not None else None
        replace_graph = getattr(backend, "replace_graph", None)
        if callable(replace_graph):
            replace_graph(
                deepcopy(list(graph_snapshot.get("nodes", []) or [])),
                deepcopy(list(graph_snapshot.get("edges", []) or [])),
            )

    def begin_transaction(self, turn_context: Any) -> None:
        temporal = getattr(turn_context, "temporal", None)
        if temporal is None:
            raise InvariantViolation("TemporalNode required -- execution invalid")
        state = getattr(turn_context, "state", None)
        service = self.turn_service
        runtime = None if service is None else getattr(service, "bot", None)
        if isinstance(state, dict):
            if runtime is not None:
                state["_save_transaction_snapshot"] = self._capture_transaction_snapshot(runtime, turn_context)
            trace_id = str(getattr(turn_context, "trace_id", "") or "")
            if trace_id:
                commit_seed = f"{trace_id}:{getattr(turn_context, 'user_input', '') or ''!s}"
                commit_id = hashlib.sha256(commit_seed.encode("utf-8")).hexdigest()[:32]
            else:
                commit_id = uuid.uuid4().hex
            state["_save_transaction"] = {
                "commit_id": commit_id,
                "trace_id": trace_id,
                "started_at": str(getattr(temporal, "wall_time", "") or ""),
                "status": "active",
            }
            state["_save_transaction_active"] = True

    def apply_mutations(self, turn_context: Any) -> None:
        temporal = getattr(turn_context, "temporal", None)
        if temporal is None:
            raise InvariantViolation("TemporalNode required -- execution invalid")
        service = self.turn_service
        runtime = None if service is None else getattr(service, "bot", None)
        if runtime is None:
            raise PersistenceFailure("SaveNode strict mode requires turn_service.bot")
        # SaveNode finalize_turn performs the canonical commit sequence in strict mode.
        # This hook exists to preserve transaction staging semantics in core.nodes.SaveNode.
        state = getattr(turn_context, "state", None)
        if isinstance(state, dict):
            state["_save_mutations_applied"] = False

    def commit_transaction(self, turn_context: Any) -> None:
        state = getattr(turn_context, "state", None)
        if isinstance(state, dict):
            transaction = dict(state.get("_save_transaction", {}) or {})
            if transaction:
                temporal = getattr(turn_context, "temporal", None)
                transaction["status"] = "committed"
                transaction["committed_at"] = str(
                    getattr(temporal, "wall_time", "") or "",
                )
                state["_save_transaction"] = transaction
                commit_id = str(transaction.get("commit_id") or "")
                state["last_commit_id"] = commit_id
                state["last_transaction_status"] = "committed"
                metadata = getattr(turn_context, "metadata", None)
                if isinstance(metadata, dict):
                    metadata["last_commit_id"] = commit_id
                    metadata["last_transaction_status"] = "committed"
                self._persist_hierarchical_memory_commit(
                    turn_context,
                    commit_id=commit_id,
                )
                self._record_merkle_anchor(turn_context, commit_id=commit_id)
            state["_save_transaction_active"] = False
            state["_save_mutations_applied"] = False

    def rollback_transaction(self, turn_context: Any) -> None:
        state = getattr(turn_context, "state", None)
        if isinstance(state, dict):
            snapshot = dict(state.get("_save_transaction_snapshot", {}) or {})
            service = self.turn_service
            runtime = None if service is None else getattr(service, "bot", None)
            if snapshot and runtime is not None:
                self._restore_transaction_snapshot(runtime, turn_context, snapshot)
            transaction = dict(state.get("_save_transaction", {}) or {})
            if transaction:
                temporal = getattr(turn_context, "temporal", None)
                transaction["status"] = "rolled_back"
                transaction["rolled_back_at"] = str(
                    getattr(temporal, "wall_time", "") or "",
                )
                state["_save_transaction"] = transaction
            state["_save_transaction_active"] = False
            state["_save_mutations_applied"] = False
            state.pop("_save_transaction_snapshot", None)
        mutation_queue = getattr(turn_context, "mutation_queue", None)
        if mutation_queue is not None:
            # Roll back queue runtime bookkeeping so failed-turn contracts are deterministic.
            reset_fn = getattr(mutation_queue, "reset_for_rollback", None)
            if callable(reset_fn):
                reset_fn()

    def final_ledger_commit(
        self,
        turn_text: str,
        mood: str,
        reply: str,
        norm_attachments: Any,
        turn_context: Any,
    ) -> tuple:
        if self.turn_service is None:
            raise PersistenceFailure(
                "SaveNode strict mode requires graph turn_service wiring in Phase 4",
            )
        return self.turn_service.finalize_user_turn(
            turn_text,
            mood,
            reply,
            norm_attachments,
            turn_context=turn_context,
        )

    def finalize_turn(self, turn_context: Any, result: Any) -> tuple:
        """Atomically commit history, maintenance, reflection, health snapshot, and persistence."""
        with ensure_execution_trace_root(
            operation="persistence_finalize_turn",
            prompt="[persistence-finalize-turn]",
            metadata={"source": "PersistenceService.finalize_turn"},
            required=True,
        ):
            return self._finalize_turn_impl(turn_context, result)

    @staticmethod
    def _finalize_fast_path_if_already_finalized(turn_context: Any, result: Any) -> tuple | None:
        # Session exit was already handled inside prepare_user_turn_async; avoid double-commit.
        if not turn_context.state.get("already_finalized"):
            return None
        if isinstance(result, tuple) and len(result) >= 2:
            return result
        return (
            str(result or ""),
            bool(turn_context.state.get("should_end", False)),
        )

    @staticmethod
    def _finalize_inputs(turn_context: Any, result: Any) -> tuple[str, str, Any, str]:
        turn_text = turn_context.state.get("turn_text") or turn_context.user_input
        mood = turn_context.state.get("mood") or "neutral"
        norm_attachments = turn_context.state.get("norm_attachments") or turn_context.attachments
        reply = result[0] if isinstance(result, tuple) else str(result or "")
        return str(turn_text or ""), str(mood or "neutral"), norm_attachments, reply

    def _finalize_validate_mutation_set(self, turn_context: Any) -> tuple[Any, Any]:
        service = self.turn_service
        if service is None:
            raise PersistenceFailure(
                "Strict mode requires graph turn_service wiring in Phase 4",
            )

        runtime = getattr(service, "bot", None)
        if runtime is None:
            raise PersistenceFailure("SaveNode strict mode requires turn_service.bot")
        if getattr(turn_context, "temporal", None) is None:
            raise InvariantViolation("TemporalNode required -- execution invalid")
        return service, runtime

    @staticmethod
    def _finalize_enforce_execution_truth_contract(service: Any, turn_context: Any) -> None:
        validator = getattr(service, "validate_execution_truth_contract", None)
        if not callable(validator):
            return
        try:
            contract = validator(turn_context, enforce=True)
            state = getattr(turn_context, "state", None)
            if isinstance(state, dict) and isinstance(contract, dict):
                state["execution_truth_contract"] = dict(contract)
        except NON_FATAL_RUNTIME_EXCEPTIONS as exc:
            raise PersistenceFailure(
                "Execution truth contract violation at SaveNode boundary",
                context={
                    "trace_id": str(getattr(turn_context, "trace_id", "") or ""),
                    "error": str(exc),
                },
            ) from exc

    @staticmethod
    def _finalize_assert_integrity_clear(turn_context: Any) -> None:
        metadata = getattr(turn_context, "metadata", None)
        state = getattr(turn_context, "state", None)
        metadata_dict = metadata if isinstance(metadata, dict) else {}
        state_dict = state if isinstance(state, dict) else {}
        refusal_state = str(state_dict.get("refusal_state") or "").strip().lower()
        if bool(metadata_dict.get("integrity_failure")) or refusal_state == "integrity_failure":
            raise IntegrityBreachError("Finalization aborted due to integrity breach markers")

    def _finalize_apply_final_graph_commit(
        self,
        runtime: Any,
        turn_context: Any,
        *,
        turn_text: str,
    ) -> None:
        self.begin_transaction(turn_context)
        self.apply_mutations(turn_context)
        # Persist behavioral ledger state before SaveNode checkpoint capture.
        self._inject_behavioral_ledger_state(runtime, turn_context, turn_text)
        self._record_relational_ledger(runtime, turn_context, turn_text)

    def _finalize_apply_ledger_commit(  # noqa: PLR0913
        self,
        runtime: Any,
        turn_context: Any,
        *,
        turn_text: str,
        mood: str,
        reply: str,
        norm_attachments: Any,
    ) -> tuple:
        finalized = self.final_ledger_commit(
            turn_text,
            mood,
            reply,
            norm_attachments,
            turn_context,
        )
        self._commit_post_finalize_side_effects(turn_context)
        return finalized

    def _finalize_trace_envelope(
        self,
        runtime: Any,
        turn_context: Any,
    ) -> None:
        if isinstance(getattr(turn_context, "metadata", None), dict):
            turn_context.metadata["async_checkpoint_contract"] = {
                "version": "async-checkpoint-order-v1",
                "ordering_model": "single-writer-monotonic-sequence",
                "enabled": bool(self._async_checkpoint_pipeline_enabled),
                "enqueue_sequence": int(self._checkpoint_enqueue_sequence),
                "drain_sequence": int(self._checkpoint_drain_sequence),
            }

        # Atomic checkpoint capture inside finalize boundary.
        checkpoint = None
        checkpoint_snapshot = getattr(turn_context, "checkpoint_snapshot", None)
        if callable(checkpoint_snapshot):
            checkpoint = checkpoint_snapshot(
                stage="save",
                status="atomic_finalize",
                error=None,
            )
            turn_context.state["_atomic_checkpoint_saved"] = True
            save_graph_checkpoint = getattr(self, "save_graph_checkpoint", None)
            if callable(save_graph_checkpoint):
                save_graph_checkpoint(checkpoint, _skip_turn_event=True)

        if self.checkpointer is not None and checkpoint is not None:
            _md = dict(getattr(turn_context, "metadata", {}) or {})
            control_plane = dict(_md.get("control_plane") or {})
            session_id = str(
                control_plane.get("session_id") or _md.get("session_id") or "default",
            )
            trace_id = str(getattr(turn_context, "trace_id", "") or "")
            manifest = dict(
                getattr(turn_context, "metadata", {}).get("determinism_manifest") or {},
            )
            checkpoint_payload = dict(checkpoint) if isinstance(checkpoint, dict) else {}
            async_allowed = bool(self._async_checkpoint_pipeline_enabled) and not bool(self.strict_mode)
            if async_allowed:
                self._enqueue_async_checkpoint(
                    session_id=session_id,
                    trace_id=trace_id,
                    checkpoint=checkpoint_payload,
                    manifest=manifest,
                )
            else:
                try:
                    self.checkpointer.save_checkpoint(
                        session_id=session_id,
                        trace_id=trace_id,
                        checkpoint=checkpoint_payload,
                        manifest=manifest,
                    )
                except NON_FATAL_RUNTIME_EXCEPTIONS as exc:
                    logger.error("PersistenceService.checkpointer save failed: %s", exc)
                    if bool(self.strict_mode):
                        raise PersistenceFailure(
                            "PersistenceService.checkpointer save failed",
                            context={"error": str(exc)},
                        ) from exc

        self._enforce_memory_authority(
            runtime,
            turn_context,
            checkpoint=dict(checkpoint) if isinstance(checkpoint, dict) else None,
        )

    def _finalize_emit_completion_event(
        self,
        runtime: Any,
        turn_context: Any,
        service: Any,
    ) -> None:
        # Completion side effects are centralized after all commit validations.
        memory_ops_started = time.perf_counter()
        self._publish_post_commit_ready(runtime, turn_context)
        memory_ops_ms = round((time.perf_counter() - memory_ops_started) * 1000, 3)
        if isinstance(getattr(turn_context, "state", None), dict):
            turn_context.state["_timing_memory_ops_ms"] = memory_ops_ms

        complete_pipeline = getattr(service, "_complete_turn_pipeline", None)
        if callable(complete_pipeline):
            complete_pipeline(should_end=False)
        self.commit_transaction(turn_context)

    def _finalize_turn_impl(self, turn_context: Any, result: Any) -> tuple:
        """Internal finalize implementation executed under an active trace context."""
        fast_path = self._finalize_fast_path_if_already_finalized(turn_context, result)
        if fast_path is not None:
            return fast_path

        # Preserve deterministic commit ordering: complete prior queued checkpoint writes
        # before starting the next SaveNode finalize boundary.
        self._drain_async_checkpoint_queue(strict_error=bool(self.strict_mode))

        turn_text, mood, norm_attachments, reply = self._finalize_inputs(turn_context, result)
        service, runtime = self._finalize_validate_mutation_set(turn_context)
        self._finalize_enforce_execution_truth_contract(service, turn_context)

        previous_temporal = getattr(runtime, "_current_turn_time_base", None)
        previous_commit_active = bool(getattr(runtime, "_graph_commit_active", False))
        finalize_started = time.perf_counter()
        try:
            self._finalize_assert_integrity_clear(turn_context)
            if previous_commit_active:
                logger.warning(
                    "PersistenceService.finalize_turn detected stale _graph_commit_active=True; "
                    "forcing fresh SaveNode boundary",
                )
            runtime._current_turn_time_base = getattr(turn_context, "temporal", None)
            runtime._graph_commit_active = True

            # 1) Validate final mutation set (preconditions + transaction gate)
            # 2) Apply final graph commit
            self._finalize_apply_final_graph_commit(
                runtime,
                turn_context,
                turn_text=turn_text,
            )

            # 3) Apply ledger commit
            finalized = self._finalize_apply_ledger_commit(
                runtime,
                turn_context,
                turn_text=turn_text,
                mood=mood,
                reply=reply,
                norm_attachments=norm_attachments,
            )

            # 4) Finalize trace envelope
            self._finalize_trace_envelope(runtime, turn_context)

            # 5) Emit completion event (single centralized post-commit side-effect stage)
            self._finalize_emit_completion_event(runtime, turn_context, service)
            return finalized
        except IntegrityBreachError:
            with contextlib.suppress(Exception):
                self.rollback_transaction(turn_context)
            raise
        except NON_FATAL_RUNTIME_EXCEPTIONS as exc:
            with contextlib.suppress(Exception):
                self.rollback_transaction(turn_context)
            logger.error(
                "PersistenceService.finalize_turn strict-mode failure: %s",
                exc,
            )
            raise PersistenceFailure(
                "PersistenceService.finalize_turn strict-mode failure",
                context={
                    "trace_id": str(getattr(turn_context, "trace_id", "") or ""),
                    "error": str(exc),
                },
            ) from exc
        finally:
            if isinstance(getattr(turn_context, "state", None), dict):
                turn_context.state["_timing_finalize_ms"] = round(
                    (time.perf_counter() - finalize_started) * 1000,
                    3,
                )
            runtime._graph_commit_active = False
            runtime._current_turn_time_base = previous_temporal

    def save_turn(self, turn_context: Any, result: Any) -> None:
        snapshot_builder = getattr(turn_context, "snapshot", None)
        if callable(snapshot_builder):
            _snap = snapshot_builder(result)
            self.persistence_manager.persist_conversation_snapshot(
                dict(_snap) if isinstance(_snap, dict) else {},
                turn_context=turn_context,
            )
            return
        self.persistence_manager.persist_conversation()

    def save_graph_checkpoint(
        self,
        checkpoint: dict[str, Any],
        _skip_turn_event: bool = False,
    ) -> None:
        try:
            with ensure_execution_trace_root(
                operation="persist_graph_checkpoint",
                prompt="[persistence-service-save-checkpoint]",
                metadata={"source": "PersistenceService.save_graph_checkpoint"},
                required=True,
            ):
                self.persistence_manager.persist_graph_checkpoint(
                    checkpoint,
                    _skip_turn_event=_skip_turn_event,
                )
        except RuntimeTraceViolation:
            trace_id = str((checkpoint or {}).get("trace_id") or "").strip()
            if not trace_id:
                logger.warning(
                    "PersistenceService.save_graph_checkpoint skipped due to null trace_id fallback",
                )
                return
            raise
        except NON_FATAL_RUNTIME_EXCEPTIONS as exc:
            logger.error("PersistenceService.save_graph_checkpoint failed: %s", exc)

    def save_turn_event(self, event: dict[str, Any]) -> None:
        try:
            self.persistence_manager.persist_turn_event(event)
        except RuntimeTraceViolation:
            trace_id = str((event or {}).get("trace_id") or "").strip()
            if not trace_id:
                logger.warning(
                    "PersistenceService.save_turn_event skipped due to null trace_id fallback",
                )
                return
            raise
        except NON_FATAL_RUNTIME_EXCEPTIONS as exc:
            logger.error("PersistenceService.save_turn_event failed: %s", exc)

    def list_turn_events(self, trace_id: str, limit: int = 0) -> list[dict[str, Any]]:
        try:
            return self.persistence_manager.list_turn_events(
                trace_id=trace_id,
                limit=limit,
            )
        except RuntimeTraceViolation:
            if not str(trace_id or "").strip():
                return []
            raise
        except NON_FATAL_RUNTIME_EXCEPTIONS as exc:
            logger.error("PersistenceService.list_turn_events failed: %s", exc)
            return []

    def list_policy_trace_events(
        self,
        *,
        trace_id: str = "",
        limit: int = 0,
    ) -> list[dict[str, Any]]:
        try:
            return self.persistence_manager.list_policy_trace_events(
                trace_id=trace_id,
                limit=limit,
            )
        except RuntimeTraceViolation:
            if not str(trace_id or "").strip():
                return []
            raise
        except NON_FATAL_RUNTIME_EXCEPTIONS as exc:
            logger.error("PersistenceService.list_policy_trace_events failed: %s", exc)
            return []

    def summarize_policy_trace_events(
        self,
        *,
        trace_id: str = "",
        limit: int = 0,
    ) -> dict[str, Any]:
        try:
            return self.persistence_manager.summarize_policy_trace_events(
                trace_id=trace_id,
                limit=limit,
            )
        except RuntimeTraceViolation:
            if not str(trace_id or "").strip():
                return {
                    "event_type": "PolicyTraceEvent",
                    "event_count": 0,
                    "policies": [],
                    "action_counts": {},
                    "latest_action": "",
                    "latest_step_name": "",
                    "latest_trace_id": "",
                }
            raise
        except NON_FATAL_RUNTIME_EXCEPTIONS as exc:
            logger.error("PersistenceService.summarize_policy_trace_events failed: %s", exc)
            return {
                "event_type": "PolicyTraceEvent",
                "event_count": 0,
                "policies": [],
                "action_counts": {},
                "latest_action": "",
                "latest_step_name": "",
                "latest_trace_id": "",
            }

    def replay_turn_events(self, trace_id: str) -> dict[str, Any]:
        try:
            return self.persistence_manager.replay_turn_events(trace_id=trace_id)
        except RuntimeTraceViolation:
            if not str(trace_id or "").strip():
                return {"trace_id": "", "events": [], "replayed_state": {}}
            raise
        except NON_FATAL_RUNTIME_EXCEPTIONS as exc:
            logger.error("PersistenceService.replay_turn_events failed: %s", exc)
            return {"trace_id": str(trace_id or ""), "events": [], "replayed_state": {}}

    def validate_replay_determinism(
        self,
        trace_id: str,
        expected_lock_hash: str = "",
    ) -> dict[str, Any]:
        try:
            return self.persistence_manager.validate_replay_determinism(
                trace_id=trace_id,
                expected_lock_hash=expected_lock_hash,
            )
        except RuntimeTraceViolation:
            if not str(trace_id or "").strip():
                return {
                    "trace_id": "",
                    "consistent": False,
                    "observed_lock_hash": "",
                    "expected_lock_hash": str(expected_lock_hash or ""),
                    "matches_expected": False,
                    "lock_hashes": [],
                }
            raise
        except NON_FATAL_RUNTIME_EXCEPTIONS as exc:
            logger.error(
                "PersistenceService.validate_replay_determinism failed: %s",
                exc,
            )
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
