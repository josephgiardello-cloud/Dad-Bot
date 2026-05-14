"""Ledger reconciliation methods extracted from ExecutionControlPlane.

Mixed into ExecutionControlPlane to reduce its LOC while keeping the
public API identical. All methods access instance attributes set by
ExecutionControlPlane.__init__.
"""
from __future__ import annotations

import hashlib
import os
from typing import Any

from dadbot.core.ledger_writer import LedgerWriter
from dadbot.core.session_store import SessionStore


class ReconciliationMixin:
    """Ledger reconciliation methods for ExecutionControlPlane."""

    def _request_has_ambiguous_inflight_effect_state(
        self,
        *,
        session_id: str,
        request_id: str,
    ) -> bool:
        return self._ledger_index.has_ambiguous_request_inflight(
            session_id=session_id,
            request_id=request_id,
        )

    def _emit_reconcile_required_event(
        self,
        *,
        session_id: str,
        trace_token: str,
        request_id: str,
        effect_id: str,
        reason: str,
    ) -> None:
        enforce = getattr(self, "_enforce_distributed_runtime_authority", None)
        if callable(enforce):
            enforce(operation="reconcile_enqueue")
        self._reconcile_queue.enqueue_required(
            session_id=session_id,
            trace_token=trace_token,
            request_id=request_id,
            effect_id=effect_id,
            reason=reason,
        )

    @staticmethod
    def _reconciliation_trace_id(*, session_id: str, request_id: str, effect_id: str) -> str:
        seed = f"{str(session_id or 'default')}|{str(request_id or '')}|{str(effect_id or '')}"
        return f"tr-reconcile-{hashlib.sha256(seed.encode('utf-8')).hexdigest()[:20]}"

    def _existing_reconcile_resolution(
        self,
        *,
        session_id: str,
        event_type: str,
        key_name: str,
        key_value: str,
    ) -> str:
        sid = str(session_id or "default").strip() or "default"
        token = str(key_value or "").strip()
        if not token:
            return ""
        for event in reversed(self.ledger.read()):
            if str(event.get("type") or "") != str(event_type or ""):
                continue
            if str(event.get("session_id") or "default") != sid:
                continue
            payload = dict(event.get("payload") or {})
            if str(payload.get(key_name) or "").strip() != token:
                continue
            return str(payload.get("resolution") or "").strip()
        return ""

    @staticmethod
    def _normalize_reconcile_mode(mode: str) -> str:
        normalized_mode = str(mode or "close_only").strip().lower() or "close_only"
        if normalized_mode not in {"close_only", "resume_eligible"}:
            raise ValueError("mode must be one of: close_only, resume_eligible")
        return normalized_mode

    @staticmethod
    def _reconcile_resolutions(normalized_mode: str) -> tuple[str, str]:
        request_resolution = (
            "closed_without_terminal"
            if normalized_mode == "close_only"
            else "resume_eligible_without_terminal"
        )
        effect_resolution = (
            "closed_without_commit"
            if normalized_mode == "close_only"
            else "resume_eligible_without_commit"
        )
        return request_resolution, effect_resolution

    def _assert_reconcile_resolution_conflicts(
        self,
        *,
        sid: str,
        rid: str,
        eid: str,
        request_resolution: str,
        effect_resolution: str,
    ) -> None:
        existing_request_resolution = self._existing_reconcile_resolution(
            session_id=sid,
            event_type="JOB_RECONCILED",
            key_name="request_id",
            key_value=rid,
        )
        if existing_request_resolution and existing_request_resolution != request_resolution:
            raise RuntimeError(
                "Conflicting reconciliation resolution for request_id "
                f"{rid!r}: existing={existing_request_resolution!r}, requested={request_resolution!r}",
            )

        existing_effect_resolution = self._existing_reconcile_resolution(
            session_id=sid,
            event_type="EFFECT_RECONCILED",
            key_name="effect_id",
            key_value=eid,
        )
        if existing_effect_resolution and existing_effect_resolution != effect_resolution:
            raise RuntimeError(
                "Conflicting reconciliation resolution for effect_id "
                f"{eid!r}: existing={existing_effect_resolution!r}, requested={effect_resolution!r}",
            )

    @staticmethod
    def _write_job_reconciled_event(
        *,
        writer: LedgerWriter,
        sid: str,
        trace_token: str,
        rid: str,
        eid: str,
        reason: str,
        request_resolution: str,
        normalized_mode: str,
    ) -> None:
        writer.write_event(
            event_type="JOB_RECONCILED",
            session_id=sid,
            trace_id=trace_token,
            kernel_step_id="control_plane.reconcile.apply",
            payload={
                "request_id": rid,
                "effect_id": eid,
                "reason": str(reason or "operator_reconcile"),
                "resolution": request_resolution,
                "mode": normalized_mode,
            },
            committed=True,
        )

    @staticmethod
    def _write_effect_reconciled_event(
        *,
        writer: LedgerWriter,
        sid: str,
        trace_token: str,
        rid: str,
        eid: str,
        reason: str,
        effect_resolution: str,
        normalized_mode: str,
    ) -> None:
        writer.write_event(
            event_type="EFFECT_RECONCILED",
            session_id=sid,
            trace_id=trace_token,
            kernel_step_id="control_plane.reconcile.apply",
            payload={
                "effect_id": eid,
                "request_id": rid,
                "reason": str(reason or "operator_reconcile"),
                "resolution": effect_resolution,
                "mode": normalized_mode,
            },
            committed=True,
        )

    def apply_reconciliation(
        self,
        *,
        session_id: str,
        request_id: str = "",
        effect_id: str = "",
        reason: str = "operator_reconcile",
        mode: str = "close_only",
    ) -> dict[str, Any]:
        """Apply deterministic reconcile closures for ambiguous request/effect inflight state."""
        enforce = getattr(self, "_enforce_distributed_runtime_authority", None)
        if callable(enforce):
            enforce(operation="reconcile_apply")
        sid = str(session_id or "default").strip() or "default"
        rid = str(request_id or "").strip()
        eid = str(effect_id or "").strip()
        if not rid and not eid:
            raise ValueError("request_id or effect_id required")
        normalized_mode = self._normalize_reconcile_mode(mode)
        request_resolution, effect_resolution = self._reconcile_resolutions(normalized_mode)
        self._assert_reconcile_resolution_conflicts(
            sid=sid,
            rid=rid,
            eid=eid,
            request_resolution=request_resolution,
            effect_resolution=effect_resolution,
        )

        trace_token = self._reconciliation_trace_id(session_id=sid, request_id=rid, effect_id=eid)
        writer = LedgerWriter(self.ledger)
        wrote_events: list[str] = []

        if rid and self._request_has_ambiguous_inflight_effect_state(session_id=sid, request_id=rid):
            self._write_job_reconciled_event(
                writer=writer,
                sid=sid,
                trace_token=trace_token,
                rid=rid,
                eid=eid,
                reason=reason,
                request_resolution=request_resolution,
                normalized_mode=normalized_mode,
            )
            wrote_events.append("JOB_RECONCILED")

        if eid and self._effect_journal.is_ambiguous(session_id=sid, effect_id=eid):
            self._write_effect_reconciled_event(
                writer=writer,
                sid=sid,
                trace_token=trace_token,
                rid=rid,
                eid=eid,
                reason=reason,
                effect_resolution=effect_resolution,
                normalized_mode=normalized_mode,
            )
            wrote_events.append("EFFECT_RECONCILED")

        self._ledger_index.refresh(force=True)
        return {
            "applied": bool(wrote_events),
            "events": wrote_events,
            "session_id": sid,
            "request_id": rid,
            "effect_id": eid,
            "reason": str(reason or "operator_reconcile"),
            "mode": normalized_mode,
        }

    def _pending_reconcile_required_entries(self) -> list[dict[str, str]]:
        return self._reconcile_queue.pending_entries()

    def _consume_reconcile_queue(self) -> dict[str, Any]:
        enabled = self._env_bool("DADBOT_AUTO_RECONCILE_ON_BOOT", True)
        max_items = self._env_int("DADBOT_RECONCILE_PASS_MAX", 64, minimum=1)
        max_rounds = self._env_int("DADBOT_RECONCILE_MAX_ROUNDS", 4, minimum=1)
        mode = str(os.environ.get("DADBOT_RECONCILE_MODE", "close_only")).strip().lower() or "close_only"
        if mode not in {"close_only", "resume_eligible"}:
            mode = "close_only"
        report = self._reconcile_queue.consume(
            enabled=enabled,
            max_items=max_items,
            max_rounds=max_rounds,
            mode=mode,
            apply=self.apply_reconciliation,
        )
        self._ledger_index.refresh(force=True)
        return dict(report)

    def boot_reconcile(self) -> dict[str, Any]:
        """Phase 3: boot reconciliation is now ledger-only via direct replay."""
        self.bootstrap()
        self._ledger_index.refresh(force=True)
        reconcile_consumer = self._consume_reconcile_queue()
        store = SessionStore(ledger=self.ledger, projection_only=True)
        events = self.ledger.read()
        store.rebuild_from_ledger(events)
        snap = store.snapshot()
        pending = list(store.pending_jobs())
        ambiguous_effects = self._ledger_index.ambiguous_effect_entries()
        ambiguous_requests = self._ledger_index.ambiguous_request_entries()
        reconcile_required = bool(ambiguous_effects or ambiguous_requests)
        return {
            "pending_jobs": pending,
            "ledger_events": len(events),
            "replay_hash": self.ledger.replay_hash(),
            "session_count": len(dict(snap.get("sessions") or {})),
            "session_snapshot_version": int(snap.get("version") or 0),
            "ledger_partitioning": self._partition_summary(events),
            "ledger_compaction": dict(self._last_compaction_report or {}),
            "execution_confluence": dict(self._last_confluence_report or {}),
            "execution_confluence_metrics": dict(self._confluence_metrics),
            "execution_lifecycle": self.lifecycle_projection.snapshot(),
            "effect_reconciliation": {
                "reconcile_required": reconcile_required,
                "ambiguous_effects": ambiguous_effects,
                "ambiguous_requests": ambiguous_requests,
                "consumer": dict(reconcile_consumer),
            },
            "ok": True,
        }
