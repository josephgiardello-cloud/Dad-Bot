from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from dadbot.core.durable_checkpoint import DurableCheckpoint
from dadbot.core.event_reducer import CanonicalEventReducer
from dadbot.core.execution_ledger import ExecutionLedger, WriteBoundaryGuard
from dadbot.core.replay_verifier import ReplayVerifier
from dadbot.core.session_store import SessionStore
from dadbot.core.snapshot_engine import SnapshotEngine


class SystemHealthChecker:
    """System-wide invariant checks for runtime consistency.

    Checks:
      - kernel-only mutation hints (static scan)
      - ledger job lifecycle completeness
      - session projection consistency against ledger replay
      - event ordering integrity
    """

    def __init__(self, *, base_path: str = ".", strict_identity: bool = False) -> None:
        self.base_path = Path(base_path)
        self._replay_verifier = ReplayVerifier()
        self._reducer = CanonicalEventReducer()
        self._snapshot_engine = SnapshotEngine(reducer=self._reducer)
        self._strict_identity = bool(strict_identity)

    @staticmethod
    def _expected_components() -> set[str]:
        return {
            "ledger_writer.write_event",
            "scheduler.drain_once",
            "graph.execute",
            "control_plane.submit_turn",
            "identity_propagation_correctness",
        }

    @staticmethod
    def _append_health_witness(ledger: ExecutionLedger, *, component: str) -> None:
        with WriteBoundaryGuard(ledger):
            ledger.write(
                {
                    "type": "EXECUTION_WITNESS",
                    "component": str(component or ""),
                    "session_id": "__health__",
                    "trace_id": "health-check",
                    "timestamp": time.time(),
                    "kernel_step_id": "system_health_checker.execution_activation",
                    "payload": {"component": str(component or "")},
                }
            )

    def check_kernel_only_mutation_enforcement(self) -> dict[str, Any]:
        offenders: list[str] = []
        for path in self.base_path.rglob("dadbot/**/*.py"):
            normalized = path.as_posix()
            if normalized.endswith("/kernel.py") or normalized.endswith("/system_health_checker.py"):
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            if ".state[" in text:
                offenders.append(normalized)
        return {
            "ok": len(offenders) == 0,
            "offenders": offenders,
            "reason": "Direct state writes found outside kernel" if offenders else "No direct state writes found",
        }

    def check_ledger_completeness(self, ledger: ExecutionLedger) -> dict[str, Any]:
        events = ledger.read()
        queued: set[str] = set()
        started: set[str] = set()
        completed_or_failed: set[str] = set()

        for event in events:
            event_type = str(event.get("type") or "")
            payload = dict(event.get("payload") or {})
            job_id = str(payload.get("job_id") or "").strip()
            if not job_id:
                continue
            if event_type == "JOB_QUEUED":
                queued.add(job_id)
            elif event_type == "JOB_STARTED":
                started.add(job_id)
            elif event_type in {"JOB_COMPLETED", "JOB_FAILED"}:
                completed_or_failed.add(job_id)

        missing_queue = sorted(started - queued)
        missing_start = sorted(completed_or_failed - started)
        ok = not missing_queue and not missing_start
        return {
            "ok": ok,
            "missing_queue_before_start": missing_queue,
            "missing_start_before_finish": missing_start,
        }

    def check_event_ordering_integrity(self, ledger: ExecutionLedger) -> dict[str, Any]:
        events = ledger.read()
        sequences = [int(event.get("sequence") or 0) for event in events]
        ordered = sequences == sorted(sequences)
        contiguous = sequences == list(range(1, len(sequences) + 1)) if sequences else True
        return {
            "ok": bool(ordered and contiguous),
            "ordered": bool(ordered),
            "contiguous": bool(contiguous),
        }

    def check_session_causal_partitioning(self, ledger: ExecutionLedger) -> dict[str, Any]:
        events = ledger.read()
        errors: list[str] = []
        session_last_index: dict[str, int] = {}
        event_session: dict[str, str] = {}

        for event in events:
            session_id = str(event.get("session_id") or "")
            event_id = str(event.get("event_id") or "")
            parent_event_id = str(event.get("parent_event_id") or "")
            session_index = int(event.get("session_index") or 0)

            if session_id:
                expected = int(session_last_index.get(session_id) or 0) + 1
                if session_index != expected:
                    errors.append(
                        f"session {session_id} has non-monotonic index {session_index} (expected {expected})"
                    )
                session_last_index[session_id] = session_index

            if event_id:
                event_session[event_id] = session_id

            if parent_event_id:
                parent_session = event_session.get(parent_event_id)
                if not parent_session:
                    errors.append(f"event {event_id or '<unknown>'} references unknown parent {parent_event_id}")
                elif parent_session != session_id:
                    errors.append(
                        f"event {event_id or '<unknown>'} leaks across sessions: parent in {parent_session}, child in {session_id}"
                    )

        return {
            "ok": len(errors) == 0,
            "errors": errors,
        }

    def check_replay_equivalence_boundary(self, ledger: ExecutionLedger) -> dict[str, Any]:
        events = ledger.read()
        session_ids = sorted({str(event.get("session_id") or "") for event in events if str(event.get("session_id") or "")})
        mismatches: list[str] = []

        for session_id in session_ids:
            session_events = [event for event in events if str(event.get("session_id") or "") == session_id]
            replayed = sorted(session_events, key=lambda event: int(event.get("sequence") or 0))
            report = self._replay_verifier.verify_equivalence(session_events, replayed)
            if not bool(report.get("ok")):
                mismatches.append(session_id)

        return {
            "ok": len(mismatches) == 0,
            "mismatched_sessions": mismatches,
            "checked_sessions": session_ids,
        }

    def check_session_store_consistency(self, *, ledger: ExecutionLedger, session_store: SessionStore) -> dict[str, Any]:
        projected = SessionStore()
        projected.rebuild_from_ledger(ledger.read())
        live_snapshot = session_store.snapshot()
        projected_snapshot = projected.snapshot()
        same_sessions = live_snapshot.get("sessions") == projected_snapshot.get("sessions")
        return {
            "ok": bool(same_sessions),
            "live_version": int(live_snapshot.get("version") or 0),
            "projected_version": int(projected_snapshot.get("version") or 0),
        }

    def check_startup_reconciliation(
        self,
        *,
        ledger: ExecutionLedger,
        session_store: SessionStore,
        checkpoint: DurableCheckpoint | None = None,
    ) -> dict[str, Any]:
        """Verify the runtime has passed the startup reconciliation gate.

        Checks:
        - If a checkpoint is provided, replay hash matches the saved head.
        - Non-empty ledger produces a non-empty session projection.
        """
        errors: list[str] = []

        if checkpoint is not None:
            chain_report = checkpoint.verify_chain_integrity()
            if not bool(chain_report.get("ok")):
                errors.extend(chain_report.get("violations") or [])

            latest = checkpoint.latest()
            if latest is not None:
                current_replay_hash = ledger.replay_hash()
                if current_replay_hash != str(latest.get("replay_hash") or ""):
                    errors.append(
                        f"Ledger replay hash diverged from checkpoint: "
                        f"expected={latest['replay_hash']!r} actual={current_replay_hash!r}"
                    )

        events = ledger.read()
        if events:
            projected = SessionStore()
            projected.rebuild_from_ledger(events)
            snap = projected.snapshot()
            if int(snap.get("version") or 0) <= 0:
                errors.append(
                    "Non-empty ledger produced empty session projection — replay is broken"
                )

        return {
            "ok": len(errors) == 0,
            "errors": errors,
            "ledger_event_count": len(events),
            "checkpoint_chain_length": (
                len(checkpoint.history()) if checkpoint is not None else 0
            ),
        }

    def check_reducer_semantic_correctness(self, ledger: ExecutionLedger) -> dict[str, Any]:
        """Step 5 — ReducerEngine semantic check.

        Validates that applying the ledger events through the canonical reducer
        produces the same state as replaying from scratch.  This is a semantic
        correctness check, not just an ordering check.

        The original approach compared sorted-vs-original event lists; this
        checks that reducer(events) == reducer(sorted(events)) — i.e. the
        execution outcome is deterministic regardless of event arrival order.
        """
        events = ledger.read()
        if not events:
            return {"ok": True, "reason": "empty ledger — nothing to check"}

        forward_state = self._reducer.reduce(events)
        sorted_state = self._reducer.reduce(
            sorted(events, key=lambda e: int(e.get("sequence") or 0))
        )

        import hashlib, json
        forward_hash = hashlib.sha256(
            json.dumps(forward_state, sort_keys=True, default=str).encode()
        ).hexdigest()
        sorted_hash = hashlib.sha256(
            json.dumps(sorted_state, sort_keys=True, default=str).encode()
        ).hexdigest()

        ok = forward_hash == sorted_hash
        return {
            "ok": ok,
            "forward_state_hash": forward_hash,
            "sorted_state_hash": sorted_hash,
            "event_count": len(events),
            "reason": (
                "reducer output is deterministic"
                if ok
                else "reducer output diverges — possible non-determinism or event corruption"
            ),
        }

    def check_snapshot_consistency(
        self,
        *,
        ledger: ExecutionLedger,
        snapshot_engine: SnapshotEngine | None = None,
    ) -> dict[str, Any]:
        """Step 8 — verify snapshot is still consistent with the current ledger."""
        engine = snapshot_engine or self._snapshot_engine
        latest = engine.latest()
        if latest is None:
            return {"ok": True, "reason": "no snapshots taken yet"}
        return engine.verify_snapshot(latest, ledger=ledger)

    def check_identity_propagation_correctness(self, ledger: ExecutionLedger) -> dict[str, Any]:
        """Validate trace/correlation identity integrity on job lifecycle events.

        Guarantees:
        - trace_id exists on lifecycle events
        - each job_id maps to exactly one trace_id and one correlation_id
        - no trace_id/correlation_id collisions across different jobs
        """
        self._append_health_witness(ledger, component="identity_propagation_correctness")
        events = ledger.read()
        lifecycle = {"JOB_SUBMITTED", "JOB_QUEUED", "JOB_STARTED", "JOB_COMPLETED", "JOB_FAILED"}

        missing_trace: list[str] = []
        missing_correlation: list[str] = []
        job_to_trace: dict[str, str] = {}
        job_to_correlation: dict[str, str] = {}
        trace_to_jobs: dict[str, set[str]] = {}
        correlation_to_jobs: dict[str, set[str]] = {}

        for event in events:
            event_type = str(event.get("type") or "")
            if event_type not in lifecycle:
                continue

            payload = dict(event.get("payload") or {})
            job_id = str(payload.get("job_id") or "").strip()
            if not job_id:
                continue

            trace_id = str(event.get("trace_id") or payload.get("trace_id") or "").strip()
            correlation_id = str(
                event.get("correlation_id")
                or payload.get("correlation_id")
                or ""
            ).strip()

            if not trace_id:
                missing_trace.append(job_id)
            if self._strict_identity and not correlation_id:
                missing_correlation.append(job_id)

            existing_trace = job_to_trace.get(job_id)
            if trace_id:
                if existing_trace is None:
                    job_to_trace[job_id] = trace_id
                elif existing_trace != trace_id:
                    missing_trace.append(job_id)
                trace_to_jobs.setdefault(trace_id, set()).add(job_id)

            existing_correlation = job_to_correlation.get(job_id)
            if correlation_id:
                if existing_correlation is None:
                    job_to_correlation[job_id] = correlation_id
                elif existing_correlation != correlation_id:
                    missing_correlation.append(job_id)
                correlation_to_jobs.setdefault(correlation_id, set()).add(job_id)

        trace_collisions = sorted(
            trace_id for trace_id, jobs in trace_to_jobs.items() if len(jobs) > 1
        )
        correlation_collisions = sorted(
            cid for cid, jobs in correlation_to_jobs.items() if len(jobs) > 1
        )

        ok = (
            len(missing_trace) == 0
            and len(missing_correlation) == 0
            and len(trace_collisions) == 0
            and len(correlation_collisions) == 0
        )
        return {
            "ok": ok,
            "strict_identity": self._strict_identity,
            "missing_trace_jobs": sorted(set(missing_trace)),
            "missing_correlation_jobs": sorted(set(missing_correlation)),
            "trace_collisions": trace_collisions,
            "correlation_collisions": correlation_collisions,
        }

    def check_execution_activation(self, ledger: ExecutionLedger) -> dict[str, Any]:
        events = ledger.read()
        executed = {
            str(event.get("component") or dict(event.get("payload") or {}).get("component") or "").strip()
            for event in events
            if str(event.get("type") or "") == "EXECUTION_WITNESS"
        }
        executed.discard("")

        expected = self._expected_components()
        missing = sorted(expected - executed)

        return {
            "ok": len(missing) == 0,
            "expected_components": sorted(expected),
            "missing_components": missing,
            "executed_components": sorted(executed),
        }

    def check_path_purity_enforcement(self, *, graph: Any | None = None) -> dict[str, Any]:
        """Verify hard boundary enforcement exists for graph execution path purity."""
        if graph is None:
            return {
                "ok": True,
                "reason": "graph not provided; path purity not checked in this run",
            }

        required_token = str(getattr(graph, "_required_execution_token", "") or "").strip()
        return {
            "ok": bool(required_token),
            "required_execution_token_present": bool(required_token),
        }

    def build_global_invariant_contract(self, checks: dict[str, Any]) -> dict[str, Any]:
        """Build a system-wide invariant contract proof over all check outputs."""
        component_names = [
            key for key, value in checks.items()
            if key != "global_invariant_contract" and isinstance(value, dict)
        ]
        failing = [
            key for key in component_names
            if not bool(checks.get(key, {}).get("ok"))
        ]

        contract = {
            "version": 1,
            "components": sorted(component_names),
            "failing_components": sorted(failing),
            "ok": len(failing) == 0,
        }
        contract_hash = hashlib.sha256(
            json.dumps(contract, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        return {
            **contract,
            "contract_hash": contract_hash,
        }

    def run_all(
        self,
        *,
        ledger: ExecutionLedger,
        session_store: SessionStore,
        graph: Any | None = None,
    ) -> dict[str, Any]:
        checks = {
            "kernel_only_mutation": self.check_kernel_only_mutation_enforcement(),
            "ledger_completeness": self.check_ledger_completeness(ledger),
            "session_projection_consistency": self.check_session_store_consistency(
                ledger=ledger,
                session_store=session_store,
            ),
            "event_ordering": self.check_event_ordering_integrity(ledger),
            "session_causal_partitioning": self.check_session_causal_partitioning(ledger),
            "replay_equivalence": self.check_replay_equivalence_boundary(ledger),
            "reducer_semantic_correctness": self.check_reducer_semantic_correctness(ledger),
            "identity_propagation_correctness": self.check_identity_propagation_correctness(ledger),
            "execution_activation": self.check_execution_activation(ledger),
            "path_purity_enforcement": self.check_path_purity_enforcement(graph=graph),
            "startup_reconciliation": self.check_startup_reconciliation(
                ledger=ledger,
                session_store=session_store,
            ),
        }
        checks["global_invariant_contract"] = self.build_global_invariant_contract(checks)
        checks["ok"] = all(bool(result.get("ok")) for result in checks.values() if isinstance(result, dict))
        return checks
