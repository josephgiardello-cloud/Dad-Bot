from __future__ import annotations

import base64
import copy
import gzip
import hashlib
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

from dadbot.contracts import DadBotContext, SupportsDadBotAccess
from dadbot.core.execution_recovery import ExecutionRecovery
from dadbot.core.execution_replay_engine import verify_terminal_state_replay_equivalence
from dadbot.core.execution_trace_context import (
    RuntimeTraceViolation,
    active_execution_trace,
    ensure_execution_trace_root,
    record_execution_step,
)


class ConversationPersistenceManager:
    """Owns conversation persistence, snapshot rehydration, and session-log writes."""

    def __init__(self, bot: DadBotContext | SupportsDadBotAccess):
        self.context = DadBotContext.from_runtime(bot)
        self.bot = self.context.bot

    def _save_commit_active(self, turn_context: Any | None = None) -> bool:
        commit_active = bool(getattr(self.bot, "_graph_commit_active", False))
        if not commit_active:
            return False
        if turn_context is None:
            return True
        active_stage = str(getattr(turn_context, "state", {}).get("_active_graph_stage") or "").strip().lower()
        return active_stage in {"save", ""}

    def _snapshot_mutation_queue(self) -> list[dict[str, Any]]:
        queue = getattr(self.bot, "_deferred_save_boundary_snapshots", None)
        if not isinstance(queue, list):
            queue = []
            self.bot._deferred_save_boundary_snapshots = queue
        return queue

    def _queue_snapshot_for_save_boundary(
        self,
        snapshot_payload: dict[str, Any],
    ) -> None:
        queue = self._snapshot_mutation_queue()
        queue.append(copy.deepcopy(dict(snapshot_payload or {})))
        if len(queue) > 32:
            del queue[:-32]

    def _active_turn_wall_time(self) -> str:
        temporal = getattr(self.bot, "_current_turn_time_base", None)
        wall_time = str(getattr(temporal, "wall_time", "") or "").strip()
        if wall_time:
            return wall_time
        return datetime.now().isoformat(timespec="seconds")

    def _active_turn_file_token(self) -> str:
        wall_time = self._active_turn_wall_time()
        compact = wall_time.replace(":", "").replace("T", "-")
        compact = compact.replace("+", "-").replace("Z", "")
        compact = compact.split(".")[0]
        return compact or datetime.now().strftime("%Y%m%d-%H%M%S")

    def _require_active_trace(self, operation: str) -> None:
        recorder = active_execution_trace()
        if recorder is None:
            raise RuntimeTraceViolation(
                f"ConversationPersistenceManager operation '{operation}' requires an active trace context",
            )
        record_execution_step(
            operation,
            payload={"layer": "persistence"},
            required=True,
        )

    def _apply_snapshot_mutations(
        self,
        snapshot_payload: dict[str, Any],
        *,
        turn_context: Any | None = None,
    ) -> None:
        chat_history = list(snapshot_payload.get("history", []))
        if chat_history and chat_history[0].get("role") == "system":
            chat_history = chat_history[1:]
        if not chat_history:
            return

        # Deterministic temporal anchors come from TurnContext; defer if absent.
        if turn_context is None:
            self._queue_snapshot_for_save_boundary(snapshot_payload)
            return

        previous_snapshot = self.bot.snapshot_session_state()
        self.bot.load_session_state_snapshot(snapshot_payload)
        try:
            self.bot.update_memory_store(chat_history, turn_context=turn_context)
            if not self.bot.LIGHT_MODE:
                self.bot.consolidate_memories(turn_context=turn_context)
            self.bot.archive_session_context(chat_history, turn_context=turn_context)
            if not self.bot.LIGHT_MODE:
                self.bot.refresh_relationship_timeline(
                    force=True,
                    turn_context=turn_context,
                )
            self.save_session_log(chat_history)
        finally:
            self.bot.load_session_state_snapshot(previous_snapshot)

    def flush_deferred_save_boundary_mutations(
        self,
        turn_context: Any | None = None,
    ) -> int:
        if not self._save_commit_active(turn_context):
            return 0
        queue = self._snapshot_mutation_queue()
        if not queue:
            return 0
        processed = 0
        pending = list(queue)
        queue.clear()
        for payload in pending:
            try:
                self._apply_snapshot_mutations(payload, turn_context=turn_context)
                processed += 1
            except Exception:
                # Re-queue failed payload for next SaveNode boundary.
                queue.append(payload)
        return processed

    def persist_conversation(self) -> None:
        with ensure_execution_trace_root(
            operation="persist_conversation",
            prompt="[conversation-persistence]",
            metadata={"source": "ConversationPersistenceManager.persist_conversation"},
            required=True,
        ):
            self._require_active_trace("persist_conversation")
            chat_history = list(self.bot.conversation_history() or [])
            if not chat_history:
                return
            # Strict mode: persist only (non-mutating). Memory/relationship mutation is SaveNode-owned.
            self.save_session_log(chat_history)

    def persist_conversation_snapshot(
        self,
        snapshot: dict,
        turn_context: Any | None = None,
    ) -> None:
        with ensure_execution_trace_root(
            operation="persist_conversation_snapshot",
            prompt="[conversation-persistence-snapshot]",
            metadata={
                "source": "ConversationPersistenceManager.persist_conversation_snapshot",
            },
            required=True,
        ):
            self._require_active_trace("persist_conversation_snapshot")
            snapshot_payload = copy.deepcopy(dict(snapshot or {}))
            chat_history = list(snapshot_payload.get("history", []))
            if chat_history and chat_history[0].get("role") == "system":
                chat_history = chat_history[1:]
            if not chat_history:
                return

            if not self._save_commit_active(turn_context):
                self._queue_snapshot_for_save_boundary(snapshot_payload)
                self.save_session_log(chat_history)
                return

            self._apply_snapshot_mutations(snapshot_payload, turn_context=turn_context)

    def save_session_log(self, history: list[dict]) -> None:
        self._require_active_trace("save_session_log")
        created_at = self._active_turn_wall_time()
        payload = {
            "created_at": created_at,
            "tenant_id": self.bot.config.tenant_id,
            "model": self.bot.config.active_model,
            "embedding_model": self.bot.config.active_embedding_model,
            "session_summary": self.bot.session_summary,
            "relationship_state": self.bot.relationship_state(),
            "history": history,
        }
        timestamp = self._active_turn_file_token()
        io_lock = getattr(self.bot, "_io_lock", None)
        if io_lock is None:
            if self.bot._tenant_document_store is not None:
                self.bot._tenant_document_store.save_session_state(
                    f"session-log:{timestamp}",
                    payload,
                )
                return
            self.bot.config.session_log_dir.mkdir(parents=True, exist_ok=True)
            session_path = self.bot.config.session_log_dir / f"session-{timestamp}.json"
            self.bot.write_json_atomically(session_path, payload, backup=False)
            return

        with io_lock:
            if self.bot._tenant_document_store is not None:
                self.bot._tenant_document_store.save_session_state(
                    f"session-log:{timestamp}",
                    payload,
                )
                return
            self.bot.SESSION_LOG_DIR.mkdir(parents=True, exist_ok=True)
            session_path = self.bot.SESSION_LOG_DIR / f"session-{timestamp}.json"
            self.bot.write_json_atomically(session_path, payload, backup=False)

    def persist_graph_checkpoint(
        self,
        checkpoint: dict,
        _skip_turn_event: bool = False,
    ) -> None:
        self._require_active_trace("persist_graph_checkpoint")
        payload = copy.deepcopy(dict(checkpoint or {}))
        if not payload:
            return
        payload["session_state"] = self.bot.snapshot_session_state()

        trace_id = str(payload.get("trace_id") or "unknown").strip() or "unknown"
        stage = str(payload.get("stage") or "stage").strip().replace(" ", "-") or "stage"
        status = str(payload.get("status") or "unknown").strip().replace(" ", "-") or "unknown"
        timestamp = self._active_turn_file_token()
        binary_payload = gzip.compress(
            pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL),
        )
        wrapped_payload = {
            "format": "gzip+pickle",
            "trace_id": trace_id,
            "stage": stage,
            "status": status,
            "created_at": self._active_turn_wall_time(),
            "payload_b64": base64.b64encode(binary_payload).decode("ascii"),
        }

        if self.bot._tenant_document_store is not None:
            checkpoint_key = f"graph-checkpoint:{trace_id}:{timestamp}:{stage}:{status}"
            self.bot._tenant_document_store.save_session_state(
                checkpoint_key,
                wrapped_payload,
            )
            self.bot._tenant_document_store.save_session_state(
                f"graph-checkpoint-latest:{trace_id}",
                wrapped_payload,
            )
            self.bot._tenant_document_store.save_session_state(
                "graph-checkpoint-latest",
                wrapped_payload,
            )
            if not _skip_turn_event:
                determinism_lock = dict(
                    (payload.get("metadata") or {}).get("determinism") or {},
                )
                self.persist_turn_event(
                    {
                        "event_type": "graph_checkpoint",
                        "trace_id": trace_id,
                        "stage": stage,
                        "status": status,
                        "determinism_lock": determinism_lock,
                        "checkpoint": payload,
                    },
                )
            return

        checkpoint_dir = self.bot.SESSION_LOG_DIR / "graph_checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / f"{timestamp}-{trace_id[:12]}-{stage}-{status}.bin"
        latest_path = checkpoint_dir / f"latest-{trace_id[:12]}.bin"
        latest_global_path = checkpoint_dir / "latest.bin"
        checkpoint_path.write_bytes(binary_payload)
        latest_path.write_bytes(binary_payload)
        latest_global_path.write_bytes(binary_payload)

        # When called directly (not from TurnGraph), write a turn event so that
        # validate_replay_determinism can fold the lock_hash from checkpoints.
        # TurnGraph calls persist_graph_checkpoint with _skip_turn_event=True and
        # emits its own turn event via save_turn_event to avoid duplicate writes.
        if not _skip_turn_event:
            determinism_lock = dict(
                (payload.get("metadata") or {}).get("determinism") or {},
            )
            self.persist_turn_event(
                {
                    "event_type": "graph_checkpoint",
                    "trace_id": trace_id,
                    "stage": stage,
                    "status": status,
                    "determinism_lock": determinism_lock,
                    "checkpoint": payload,
                },
            )

    def _turn_event_dir(self) -> Path:
        event_dir = self.bot.SESSION_LOG_DIR / "turn_events"
        event_dir.mkdir(parents=True, exist_ok=True)
        return event_dir

    def _turn_event_path(self, trace_id: str) -> Path:
        normalized = str(trace_id or "unknown").strip() or "unknown"
        return self._turn_event_dir() / f"{normalized[:40]}.jsonl"

    def _snapshot_path(self, trace_id: str) -> Path:
        normalized = str(trace_id or "unknown").strip() or "unknown"
        return self._turn_event_dir() / f"snapshot-{normalized[:40]}.json"

    def _compute_next_sequence(self, trace_id: str) -> int:
        events = self.list_turn_events(trace_id=trace_id)
        if not events:
            return 1
        return int(events[-1].get("sequence") or len(events)) + 1

    def persist_turn_event(self, event: dict[str, Any]) -> None:
        self._require_active_trace("persist_turn_event")
        payload = copy.deepcopy(dict(event or {}))
        if not payload:
            return
        trace_id = str(payload.get("trace_id") or "unknown").strip() or "unknown"
        payload["trace_id"] = trace_id
        sequence = self._compute_next_sequence(trace_id)
        payload.setdefault("sequence", sequence)
        event_id_seed = f"{trace_id}:{int(payload.get('sequence') or sequence)}"
        payload.setdefault(
            "event_id",
            hashlib.sha256(event_id_seed.encode()).hexdigest()[:16],
        )
        payload.setdefault("occurred_at", self._active_turn_wall_time())

        if self.bot._tenant_document_store is not None:
            key = f"turn-events:{trace_id}"
            existing = self.bot._tenant_document_store.load_session_state(key)
            events = list(existing) if isinstance(existing, list) else []
            events.append(payload)
            self.bot._tenant_document_store.save_session_state(key, events)
            self.bot._tenant_document_store.save_session_state(
                f"turn-events-latest:{trace_id}",
                payload,
            )
        else:
            event_path = self._turn_event_path(trace_id)
            line = json.dumps(payload, sort_keys=True, ensure_ascii=True)
            with event_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

        # Snapshot compaction every 25 events: replay can start from latest snapshot.
        compact_sequence = int(payload.get("sequence") or 0)
        if compact_sequence > 0 and compact_sequence % 25 == 0:
            replay = self.replay_turn_events(trace_id=trace_id)
            snapshot_payload = {
                "trace_id": trace_id,
                "created_at": self._active_turn_wall_time(),
                "last_sequence": compact_sequence,
                "phase": replay.get("phase"),
                "state": replay.get("replayed_state", {}),
                "metadata": replay.get("replayed_metadata", {}),
            }
            if self.bot._tenant_document_store is not None:
                self.bot._tenant_document_store.save_session_state(
                    f"turn-snapshot-latest:{trace_id}",
                    snapshot_payload,
                )
            else:
                snapshot_path = self._snapshot_path(trace_id)
                snapshot_path.write_text(
                    json.dumps(snapshot_payload, sort_keys=True, ensure_ascii=True),
                    encoding="utf-8",
                )

    def list_turn_events(self, trace_id: str, limit: int = 0) -> list[dict[str, Any]]:
        self._require_active_trace("list_turn_events")
        normalized = str(trace_id or "").strip()
        if not normalized:
            return []

        events: list[dict[str, Any]] = []
        if self.bot._tenant_document_store is not None:
            stored = self.bot._tenant_document_store.load_session_state(
                f"turn-events:{normalized}",
            )
            if isinstance(stored, list):
                events = [item for item in stored if isinstance(item, dict)]
        else:
            event_path = self._turn_event_path(normalized)
            if event_path.exists():
                for raw_line in event_path.read_text(encoding="utf-8").splitlines():
                    line = str(raw_line or "").strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                    except Exception:
                        continue
                    if isinstance(item, dict):
                        events.append(item)

        events.sort(key=lambda item: int(item.get("sequence") or 0))
        if limit and limit > 0:
            return events[-limit:]
        return events

    @staticmethod
    def _fold_events(
        events: list[dict[str, Any]],
    ) -> tuple[dict[str, Any], dict[str, Any], str, dict[str, Any]]:
        replayed_state: dict[str, Any] = {}
        replayed_metadata: dict[str, Any] = {}
        phase = "PLAN"
        seen_lock_hashes: set[str] = set()
        latest_execution_identity: dict[str, Any] = {}
        for event in events:
            if not isinstance(event, dict):
                continue
            determinism_lock = event.get("determinism_lock")
            if isinstance(determinism_lock, dict):
                lock_hash = str(determinism_lock.get("lock_hash") or "").strip()
                if lock_hash:
                    seen_lock_hashes.add(lock_hash)
            checkpoint = event.get("checkpoint")
            if isinstance(checkpoint, dict):
                state = checkpoint.get("state")
                metadata = checkpoint.get("metadata")
                if isinstance(state, dict):
                    replayed_state.update(copy.deepcopy(state))
                if isinstance(metadata, dict):
                    replayed_metadata.update(copy.deepcopy(metadata))
                replayed_determinism = metadata.get("determinism") if isinstance(metadata, dict) else None
                if isinstance(replayed_determinism, dict):
                    lock_hash = str(replayed_determinism.get("lock_hash") or "").strip()
                    if lock_hash:
                        seen_lock_hashes.add(lock_hash)
                checkpoint_phase = str(checkpoint.get("phase") or "").strip()
                if checkpoint_phase:
                    phase = checkpoint_phase
            if str(event.get("event_type") or "") == "execution_identity":
                identity_data = event.get("identity")
                if isinstance(identity_data, dict):
                    latest_execution_identity = dict(identity_data)
            event_phase = str(event.get("phase") or "").strip()
            if event_phase:
                phase = event_phase

        determinism_summary = {
            "lock_hash": next(iter(seen_lock_hashes)) if len(seen_lock_hashes) == 1 else "",
            "lock_hashes": sorted(seen_lock_hashes),
            "consistent": len(seen_lock_hashes) <= 1,
            "execution_identity": latest_execution_identity,
            "execution_fingerprint": str(
                latest_execution_identity.get("fingerprint") or "",
            ),
        }
        return replayed_state, replayed_metadata, phase, determinism_summary

    def replay_turn_events(self, trace_id: str) -> dict[str, Any]:
        self._require_active_trace("replay_turn_events")
        normalized = str(trace_id or "").strip()
        events = self.list_turn_events(trace_id=normalized)
        replayed_state, replayed_metadata, phase, determinism = self._fold_events(
            events,
        )
        return {
            "trace_id": normalized,
            "events": events,
            "phase": phase,
            "replayed_state": replayed_state,
            "replayed_metadata": replayed_metadata,
            "determinism": determinism,
            "event_count": len(events),
        }

    def validate_replay_determinism(
        self,
        trace_id: str,
        expected_lock_hash: str = "",
        *,
        expected_terminal_state: dict[str, Any] | None = None,
        expected_execution_trace_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._require_active_trace("validate_replay_determinism")
        replay = self.replay_turn_events(trace_id=trace_id)
        determinism = dict(replay.get("determinism") or {})
        observed_hash = str(determinism.get("lock_hash") or "").strip()
        expected_hash = str(expected_lock_hash or "").strip()
        matches_expected = True
        if expected_hash:
            matches_expected = observed_hash == expected_hash

        terminal_equivalence: dict[str, Any] = {}
        repaired_trace_context: dict[str, Any] = {}
        effective_terminal_state = (
            expected_terminal_state
            if isinstance(expected_terminal_state, dict)
            else dict(
                (replay.get("replayed_metadata") or {}).get("terminal_state") or {},
            )
        )
        effective_trace_context = (
            expected_execution_trace_context
            if isinstance(expected_execution_trace_context, dict)
            else dict(
                (replay.get("replayed_metadata") or {}).get("execution_trace_context") or {},
            )
        )
        if (
            isinstance(effective_terminal_state, dict)
            and isinstance(effective_trace_context, dict)
            and effective_terminal_state
            and effective_trace_context
        ):
            repaired_trace_context = ExecutionRecovery.repair_partial_trace_context(
                effective_trace_context,
            )
            terminal_equivalence = verify_terminal_state_replay_equivalence(
                terminal_state_seed=effective_terminal_state,
                execution_trace_context=repaired_trace_context,
                enforce_dag_equivalence=True,
            )
        return {
            "trace_id": str(trace_id or "").strip(),
            "consistent": bool(determinism.get("consistent", True)),
            "observed_lock_hash": observed_hash,
            "expected_lock_hash": expected_hash,
            "matches_expected": matches_expected,
            "lock_hashes": list(determinism.get("lock_hashes") or []),
            "execution_identity": dict(determinism.get("execution_identity") or {}),
            "execution_fingerprint": str(
                determinism.get("execution_fingerprint") or "",
            ),
            "terminal_state_equivalence": terminal_equivalence,
            "trace_repair_applied": bool(
                isinstance(effective_trace_context, dict) and effective_trace_context != repaired_trace_context,
            ),
            "verification_path": "posthoc",
        }

    def load_latest_graph_checkpoint(self, trace_id: str = "") -> dict | None:
        self._require_active_trace("load_latest_graph_checkpoint")
        encoded_payload = None
        normalized_trace_id = str(trace_id or "").strip()
        if self.bot._tenant_document_store is not None:
            lookup_key = (
                f"graph-checkpoint-latest:{normalized_trace_id}" if normalized_trace_id else "graph-checkpoint-latest"
            )
            encoded_payload = self.bot._tenant_document_store.load_session_state(
                lookup_key,
            )
        else:
            checkpoint_dir = self.bot.SESSION_LOG_DIR / "graph_checkpoints"
            checkpoint_path = checkpoint_dir / (
                f"latest-{normalized_trace_id[:12]}.bin" if normalized_trace_id else "latest.bin"
            )
            if checkpoint_path.exists():
                try:
                    return pickle.loads(gzip.decompress(checkpoint_path.read_bytes()))
                except Exception:
                    return None
        if not isinstance(encoded_payload, dict):
            return None
        payload_b64 = str(encoded_payload.get("payload_b64") or "").strip()
        if not payload_b64:
            return None
        try:
            return pickle.loads(gzip.decompress(base64.b64decode(payload_b64)))
        except Exception:
            return None

    def resume_graph_checkpoint(self, trace_id: str = "") -> dict | None:
        self._require_active_trace("resume_graph_checkpoint")
        checkpoint = self.load_latest_graph_checkpoint(trace_id=trace_id)
        if not isinstance(checkpoint, dict):
            return None
        session_state = checkpoint.get("session_state")
        if isinstance(session_state, dict):
            self.bot.load_session_state_snapshot(session_state)
        return checkpoint


__all__ = ["ConversationPersistenceManager"]
