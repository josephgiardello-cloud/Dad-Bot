from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from copy import deepcopy
from typing import Any

from dadbot.core.capability_audit_runner import (
    CAPABILITY_AUDIT_EVENT_TYPE,
    build_capability_audit_event_payload,
    build_runtime_capability_audit_report,
)
from dadbot.core.execution_context import (
    RuntimeTraceViolation,
    ensure_execution_trace_root,
)
from dadbot.core.kernel_locks import KernelEventTotalityLock
from dadbot.core.kernel_mutation_gate import apply_event, emit_event
from dadbot.core.graph import (
    FatalTurnError,
    LedgerMutationOp,
    MemoryMutationOp,
    MutationIntent,
    MutationKind,
)
from dadbot.core.merkle_anchor import append_leaf_and_anchor
from dadbot.core.persistence import AbstractCheckpointer
from dadbot.core.post_commit_events import PostCommitEvent
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

    @staticmethod
    def _stable_hash(payload: Any) -> str:
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str).encode(
                "utf-8",
            ),
        ).hexdigest()

    def _record_merkle_anchor(self, turn_context: Any, *, commit_id: str) -> None:
        runtime = getattr(getattr(self.turn_service, "bot", None), "config", None)
        enabled = bool(getattr(runtime, "merkle_anchor_enabled", True))
        if not enabled:
            return

        metadata = getattr(turn_context, "metadata", None)
        control_plane = dict(metadata.get("control_plane") or {}) if isinstance(metadata, dict) else {}
        session_id = str(control_plane.get("session_id") or "default")
        trace_id = str(getattr(turn_context, "trace_id", "") or "")
        temporal = getattr(turn_context, "temporal", None)
        occurred_at = str(getattr(temporal, "wall_time", "") or "")
        payload = {
            "session_id": session_id,
            "trace_id": trace_id,
            "commit_id": str(commit_id or ""),
            "occurred_at": occurred_at,
            "state_hash": self._stable_hash(getattr(turn_context, "state", {}) or {}),
            "metadata_hash": self._stable_hash(
                getattr(turn_context, "metadata", {}) or {},
            ),
        }
        leaves = self._merkle_session_leaves.setdefault(session_id, [])
        anchor = append_leaf_and_anchor(leaves, payload)
        if isinstance(metadata, dict):
            metadata_event = emit_event(
                "MUTATION_EVENT",
                {"op": "dict_set", "key": "merkle_anchor", "value": dict(anchor)},
                source="PersistenceService._record_merkle_anchor",
            )
            turn_context.metadata = apply_event(
                metadata_event,
                metadata,
                lambda state, evt: {**state, str(evt.payload.get("key") or ""): evt.payload.get("value")},
            )
        if isinstance(getattr(turn_context, "state", None), dict):
            state_event = emit_event(
                "MUTATION_EVENT",
                {"op": "dict_set", "key": "merkle_anchor", "value": dict(anchor)},
                source="PersistenceService._record_merkle_anchor",
            )
            turn_context.state = apply_event(
                state_event,
                turn_context.state,
                lambda state, evt: {**state, str(evt.payload.get("key") or ""): evt.payload.get("value")},
            )
        self.save_turn_event(
            {
                "event_type": "merkle_anchor_commit",
                "trace_id": trace_id,
                "occurred_at": occurred_at,
                "stage": "save",
                "status": "after",
                "payload": {
                    "session_id": session_id,
                    "commit_id": str(commit_id or ""),
                    **anchor,
                },
            },
        )

    def set_checkpointer(self, checkpointer: AbstractCheckpointer | None) -> None:
        self.checkpointer = checkpointer

    @staticmethod
    def _call_nonfatal(callable_obj: Any, *args: Any, **kwargs: Any) -> Any:
        if not callable(callable_obj):
            return None
        try:
            return callable_obj(*args, **kwargs)
        except Exception as exc:
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
        raise RuntimeError(
            "PersistenceService._apply_memory_decay is not allowed; "
            "run decay via post-commit worker or maintenance service"
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
            raise RuntimeError("SaveNode strict mode requires runtime.memory")

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

    def _drain_mutation_queue(self, runtime: Any, turn_context: Any) -> None:
        mutation_queue = getattr(turn_context, "mutation_queue", None)
        if mutation_queue is None:
            return

        service = self.turn_service

        def _resolve_event_tap() -> Any:
            direct = getattr(runtime, "event_tap", None) or getattr(runtime, "_event_tap", None)
            if direct is not None:
                return direct
            services = getattr(runtime, "services", None)
            return getattr(services, "event_tap", None)

        def _dispatch_mutation_intent(intent: Any) -> None:
            if not isinstance(intent, MutationIntent):
                raise RuntimeError(
                    f"MutationQueue received non-MutationIntent payload: {type(intent).__name__}",
                )
            intent_type = intent.type
            payload = dict(intent.payload or {})
            source = str(intent.source or "")
            trace_id = str(getattr(turn_context, "trace_id", "") or "").strip()
            source_tag = str(getattr(intent, "source", "") or "")
            witness: dict[str, Any] = {}
            if trace_id:
                tap = _resolve_event_tap()
                emit = getattr(tap, "emit", None)
                # Enforce strictly only when an event tap is active for this runtime.
                if callable(emit):
                    try:
                        witness = KernelEventTotalityLock.require_event_witness(
                            run_id=trace_id,
                            source=source_tag,
                        )
                    except RuntimeError as exc:
                        emit(
                            "MUTATION_EVENT_TOTALITY_VIOLATION",
                            run_id=trace_id,
                            source=source_tag,
                            error=str(exc),
                        )
                        raise
            if intent_type is MutationKind.GRAPH:
                memory_manager = getattr(runtime, "memory_manager", None)
                graph_manager = getattr(memory_manager, "graph_manager", None) if memory_manager else None
                if graph_manager is None:
                    raise RuntimeError(
                        f"MutationIntent(type=graph, source={source!r}): graph_manager unavailable",
                    )
                _fn = getattr(graph_manager, "apply_mutation", None)
                if callable(_fn):
                    _fn(payload, turn_context=turn_context)
                    return
                raise RuntimeError(
                    f"MutationIntent(type=graph, source={source!r}): graph_manager.apply_mutation not callable",
                )

            if intent_type is MutationKind.LEDGER:
                op = str(payload.get("op") or "").strip().lower()
                if op == LedgerMutationOp.APPEND_HISTORY.value:
                    entry = dict(payload.get("entry") or {})
                    with runtime._session_lock:
                        runtime.history.append(entry)
                    return
                if op == LedgerMutationOp.RECORD_TURN_STATE.value:
                    mood = str(payload.get("mood") or "neutral")
                    should_offer_daily_checkin = bool(
                        payload.get("should_offer_daily_checkin", False),
                    )
                    with runtime._session_lock:
                        runtime.session_moods.append(mood)
                        runtime._pending_daily_checkin_context = should_offer_daily_checkin
                    return
                if op == LedgerMutationOp.SYNC_THREAD_SNAPSHOT.value:
                    runtime.sync_active_thread_snapshot()
                    return
                if op == LedgerMutationOp.CLEAR_TURN_CONTEXT.value:
                    with runtime._session_lock:
                        runtime._pending_daily_checkin_context = False
                        runtime._active_tool_observation_context = None
                    return
                if op == LedgerMutationOp.SCHEDULE_MAINTENANCE.value:
                    turn_text = str(payload.get("turn_text") or "")
                    mood = payload.get("mood")
                    if not bool(getattr(runtime, "LIGHT_MODE", False)):
                        runtime.schedule_post_turn_maintenance(turn_text, mood)
                        append_step = getattr(
                            service,
                            "_append_turn_pipeline_step",
                            None,
                        )
                        if callable(append_step):
                            append_step(
                                "schedule_maintenance",
                                detail="queued post-turn maintenance",
                            )
                    else:
                        append_step = getattr(
                            service,
                            "_append_turn_pipeline_step",
                            None,
                        )
                        if callable(append_step):
                            append_step(
                                "schedule_maintenance",
                                status="skipped",
                                detail="light mode skips maintenance",
                            )
                    return
                if op == LedgerMutationOp.HEALTH_SNAPSHOT.value:
                    runtime.current_runtime_health_snapshot(
                        force=True,
                        log_warnings=True,
                        persist=True,
                    )
                    append_step = getattr(service, "_append_turn_pipeline_step", None)
                    if callable(append_step):
                        append_step(
                            "health_snapshot",
                            detail="refreshed runtime health snapshot",
                        )
                    return
                if op == LedgerMutationOp.CAPABILITY_AUDIT_EVENT.value:
                    # Non-authoritative observability write: never block turn correctness.
                    try:
                        stage_order = [
                            str(getattr(trace, "stage", "") or "")
                            for trace in list(
                                getattr(turn_context, "stage_traces", []) or [],
                            )
                        ]
                        if "save" not in [s.strip().lower() for s in stage_order]:
                            stage_order = [*stage_order, "save"]

                        report = build_runtime_capability_audit_report(
                            turn_context,
                            stage_order=stage_order,
                            failed=False,
                        )
                        report_payload = report.to_dict()
                        if isinstance(getattr(turn_context, "state", None), dict):
                            turn_context.state["capability_audit_report"] = report_payload
                        if isinstance(getattr(turn_context, "metadata", None), dict):
                            turn_context.metadata["capability_audit_report"] = dict(
                                report_payload,
                            )

                        event_payload = build_capability_audit_event_payload(
                            report,
                            scenario=str(payload.get("scenario") or "runtime_turn"),
                        )
                        if hasattr(turn_context, "event_sequence"):
                            turn_context.event_sequence += 1
                            sequence = int(turn_context.event_sequence)
                        else:
                            sequence = 0

                        trace_id = str(
                            getattr(turn_context, "trace_id", "") or "unknown",
                        )
                        phase_value = str(
                            getattr(getattr(turn_context, "phase", None), "value", "") or "",
                        )
                        occurred_at = ""
                        temporal = getattr(turn_context, "temporal", None)
                        if temporal is not None:
                            occurred_at = str(getattr(temporal, "wall_time", "") or "")

                        self.save_turn_event(
                            {
                                "event_type": CAPABILITY_AUDIT_EVENT_TYPE,
                                "trace_id": trace_id,
                                "sequence": sequence,
                                "occurred_at": occurred_at,
                                "stage": "save",
                                "status": "after",
                                "phase": phase_value,
                                "payload": event_payload,
                            },
                        )

                        control_plane = getattr(
                            getattr(runtime, "turn_orchestrator", None),
                            "control_plane",
                            None,
                        )
                        ledger_writer = getattr(control_plane, "ledger_writer", None)
                        write_event = getattr(ledger_writer, "write_event", None)
                        if callable(write_event):
                            session_id = str(
                                (getattr(turn_context, "metadata", {}) or {}).get("control_plane", {}).get("session_id")
                                or "default",
                            )
                            write_event(
                                event_type=CAPABILITY_AUDIT_EVENT_TYPE,
                                session_id=session_id,
                                trace_id=trace_id,
                                kernel_step_id="save_node.capability_audit",
                                payload=event_payload,
                                committed=False,
                            )
                    except Exception as exc:
                        logger.warning(
                            "PersistenceService capability audit persistence failed (non-fatal): %s",
                            exc,
                        )
                    return
                raise RuntimeError(
                    f"MutationIntent(type=ledger, source={source!r}): unsupported op={op!r}",
                )

            raise RuntimeError(
                f"MutationIntent: unknown type={intent_type!r} source={source!r}",
            )

        try:
            mutation_queue.drain(
                _dispatch_mutation_intent,
                hard_fail_on_error=True,
                transactional=True,
            )
        except TypeError as exc:
            # Backward compatibility for tests/wrappers that monkeypatch
            # MutationQueue.drain with a legacy two-parameter signature.
            if "unexpected keyword argument 'transactional'" not in str(exc):
                raise
            mutation_queue.drain(
                _dispatch_mutation_intent,
                hard_fail_on_error=True,
            )
        if not mutation_queue.is_empty():
            pending = mutation_queue.size()
            raise FatalTurnError(
                "Mutation queue not fully drained"
                f" (pending={pending}, trace_id={getattr(turn_context, 'trace_id', '')!r})",
            )

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

    @staticmethod
    def _build_hierarchical_memory_payload(turn_context: Any) -> dict[str, Any]:
        state = getattr(turn_context, "state", None)
        metadata = getattr(turn_context, "metadata", None)
        state_dict = state if isinstance(state, dict) else {}
        metadata_dict = metadata if isinstance(metadata, dict) else {}

        return {
            "recent_buffer": list(state_dict.get("memory_recent_buffer") or []),
            "rolling_summary": str(state_dict.get("memory_rolling_summary") or ""),
            "structured_memory": dict(state_dict.get("memory_structured") or {}),
            "full_history_id": state_dict.get("memory_full_history_id"),
            "token_counts": {
                "recent": int(metadata_dict.get("recent_tokens", 0) or 0),
                "summary": int(metadata_dict.get("summary_tokens", 0) or 0),
                "structured": int(metadata_dict.get("structured_tokens", 0) or 0),
                "total": int(metadata_dict.get("context_total_tokens", 0) or 0),
            },
        }

    def _persist_hierarchical_memory_commit(
        self,
        turn_context: Any,
        *,
        commit_id: str,
    ) -> None:
        state = getattr(turn_context, "state", None)
        metadata = getattr(turn_context, "metadata", None)
        if not isinstance(state, dict):
            return

        memory_payload = self._build_hierarchical_memory_payload(turn_context)
        state["hierarchical_memory_payload"] = dict(memory_payload)
        if isinstance(metadata, dict):
            metadata["hierarchical_memory_payload"] = dict(memory_payload)

        trace_id = str(getattr(turn_context, "trace_id", "") or "")
        phase_value = str(
            getattr(getattr(turn_context, "phase", None), "value", "") or "",
        )
        occurred_at = ""
        temporal = getattr(turn_context, "temporal", None)
        if temporal is not None:
            occurred_at = str(getattr(temporal, "wall_time", "") or "")

        self.save_turn_event(
            {
                "event_type": "hierarchical_memory_commit",
                "trace_id": trace_id,
                "occurred_at": occurred_at,
                "stage": "save",
                "status": "after",
                "phase": phase_value,
                "payload": {
                    "commit_id": str(commit_id or ""),
                    "trace_id": trace_id,
                    "memory": memory_payload,
                },
            },
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
            raise RuntimeError(
                "SaveNode strict mode requires an attached turn_service runtime",
            )

        # --- Drain MutationQueue FIRST at the canonical SaveNode commit boundary ---
        # Every mutation queued outside this boundary (deferred from earlier stages or
        # direct-path callers) must execute here. Any failure is a hard fail — nothing
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
            raise RuntimeError(
                "SaveNode strict mode requires relationship_projector.materialize_projection",
            )
        materialize_projection(turn_context=turn_context)

        memory_manager = getattr(runtime, "memory_manager", None)
        graph_manager = getattr(memory_manager, "graph_manager", None) if memory_manager is not None else None
        sync_graph_store = getattr(graph_manager, "sync_graph_store", None)
        if not callable(sync_graph_store):
            raise RuntimeError(
                "SaveNode strict mode requires memory_graph_manager.sync_graph_store",
            )
        graph_sync_started = time.perf_counter()
        sync_graph_store(turn_context=turn_context)
        graph_sync_ms = round((time.perf_counter() - graph_sync_started) * 1000, 3)
        if isinstance(getattr(turn_context, "state", None), dict):
            turn_context.state["_timing_graph_sync_ms"] = graph_sync_ms

        # Publish post-commit readiness only after the full strict commit
        # sequence succeeds, so failed commits cannot leak readiness events.
        memory_ops_started = time.perf_counter()
        self._publish_post_commit_ready(runtime, turn_context)
        memory_ops_ms = round((time.perf_counter() - memory_ops_started) * 1000, 3)
        if isinstance(getattr(turn_context, "state", None), dict):
            turn_context.state["_timing_memory_ops_ms"] = memory_ops_ms

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
            raise RuntimeError("TemporalNode required — execution invalid")
        state = getattr(turn_context, "state", None)
        service = self.turn_service
        runtime = None if service is None else getattr(service, "bot", None)
        if isinstance(state, dict):
            if runtime is not None:
                state["_save_transaction_snapshot"] = self._capture_transaction_snapshot(runtime, turn_context)
            trace_id = str(getattr(turn_context, "trace_id", "") or "")
            state["_save_transaction"] = {
                "commit_id": uuid.uuid4().hex,
                "trace_id": trace_id,
                "started_at": str(getattr(temporal, "wall_time", "") or ""),
                "status": "active",
            }
            state["_save_transaction_active"] = True

    def apply_mutations(self, turn_context: Any) -> None:
        temporal = getattr(turn_context, "temporal", None)
        if temporal is None:
            raise RuntimeError("TemporalNode required — execution invalid")
        service = self.turn_service
        runtime = None if service is None else getattr(service, "bot", None)
        if runtime is None:
            raise RuntimeError("SaveNode strict mode requires turn_service.bot")
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
            raise RuntimeError(
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
        # Session exit was already handled inside prepare_user_turn_async â€” skip double-commit.
        if turn_context.state.get("already_finalized"):
            if isinstance(result, tuple) and len(result) >= 2:
                return result
            return (
                str(result or ""),
                bool(turn_context.state.get("should_end", False)),
            )

        turn_text = turn_context.state.get("turn_text") or turn_context.user_input
        mood = turn_context.state.get("mood") or "neutral"
        norm_attachments = turn_context.state.get("norm_attachments") or turn_context.attachments
        reply = result[0] if isinstance(result, tuple) else str(result or "")

        service = self.turn_service
        if service is None:
            raise RuntimeError(
                "Strict mode requires graph turn_service wiring in Phase 4",
            )

        runtime = getattr(service, "bot", None)
        if runtime is None:
            raise RuntimeError("SaveNode strict mode requires turn_service.bot")
        if getattr(turn_context, "temporal", None) is None:
            raise RuntimeError("TemporalNode required — execution invalid")

        previous_temporal = getattr(runtime, "_current_turn_time_base", None)
        previous_commit_active = bool(getattr(runtime, "_graph_commit_active", False))
        finalize_started = time.perf_counter()
        try:
            if previous_commit_active:
                logger.warning(
                    "PersistenceService.finalize_turn detected stale _graph_commit_active=True; "
                    "forcing fresh SaveNode boundary",
                )
            runtime._current_turn_time_base = getattr(turn_context, "temporal", None)
            runtime._graph_commit_active = True

            # Strict sequence: finalize queues ledger intents; then one canonical SaveNode commit.
            finalized = self.final_ledger_commit(
                turn_text,
                mood,
                reply,
                norm_attachments,
                turn_context,
            )
            self._commit_post_finalize_side_effects(turn_context)

            # Atomic checkpoint capture: emit from inside finalize_turn boundary so
            # hash-chain state is committed with the same save transaction.
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
                try:
                    self.checkpointer.save_checkpoint(
                        session_id=session_id,
                        trace_id=trace_id,
                        checkpoint=dict(checkpoint) if isinstance(checkpoint, dict) else {},
                        manifest=manifest,
                    )
                except Exception as exc:
                    # Non-fatal by default; strict mode escalates.
                    logger.error("PersistenceService.checkpointer save failed: %s", exc)
                    if bool(self.strict_mode):
                        raise

            complete_pipeline = getattr(service, "_complete_turn_pipeline", None)
            if callable(complete_pipeline):
                complete_pipeline(should_end=False)
            return finalized
        except Exception as exc:
            logger.error(
                "PersistenceService.finalize_turn strict-mode failure: %s",
                exc,
            )
            raise
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
            raise
        except Exception as exc:
            logger.error("PersistenceService.save_graph_checkpoint failed: %s", exc)

    def save_turn_event(self, event: dict[str, Any]) -> None:
        try:
            self.persistence_manager.persist_turn_event(event)
        except RuntimeTraceViolation:
            raise
        except Exception as exc:
            logger.error("PersistenceService.save_turn_event failed: %s", exc)

    def list_turn_events(self, trace_id: str, limit: int = 0) -> list[dict[str, Any]]:
        try:
            return self.persistence_manager.list_turn_events(
                trace_id=trace_id,
                limit=limit,
            )
        except RuntimeTraceViolation:
            raise
        except Exception as exc:
            logger.error("PersistenceService.list_turn_events failed: %s", exc)
            return []

    def replay_turn_events(self, trace_id: str) -> dict[str, Any]:
        try:
            return self.persistence_manager.replay_turn_events(trace_id=trace_id)
        except RuntimeTraceViolation:
            raise
        except Exception as exc:
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
            raise
        except Exception as exc:
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
