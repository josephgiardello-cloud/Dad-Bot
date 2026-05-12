"""Ledger compaction helpers extracted from ExecutionControlPlane."""
from __future__ import annotations

import gzip
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

from dadbot.core.compaction import ArchiveTier, CompactionPolicy, EventCompactor
from dadbot.core.semantic_primitives import hash as semantic_hash

logger = logging.getLogger(__name__)


class CompactionMixin:
    @staticmethod
    def _stable_hash(payload: Any) -> str:
        return semantic_hash(payload)

    def _job_trace_events(self, job) -> list[dict[str, Any]]:
        events = list(self.ledger.read())
        trace_id = str(job.trace_id or "").strip()
        job_id = str(job.job_id or "").strip()
        filtered = [
            dict(event)
            for event in events
            if (
                str(event.get("trace_id") or "").strip() == trace_id
                or str(dict(event.get("payload") or {}).get("job_id") or "").strip() == job_id
            )
        ]
        return sorted(filtered, key=lambda item: int(item.get("sequence") or 0))

    @staticmethod
    def _event_stream_digest(events: list[dict[str, Any]]) -> str:
        canonical = [
            {
                "sequence": int(event.get("sequence") or 0),
                "type": str(event.get("type") or ""),
                "trace_id": str(event.get("trace_id") or ""),
                "session_id": str(event.get("session_id") or ""),
                "kernel_step_id": str(event.get("kernel_step_id") or ""),
                "payload_hash": hashlib.sha256(
                    json.dumps(dict(event.get("payload") or {}), sort_keys=True, default=str).encode("utf-8"),
                ).hexdigest(),
            }
            for event in list(events or [])
        ]
        return hashlib.sha256(
            json.dumps(canonical, sort_keys=True, default=str).encode("utf-8"),
        ).hexdigest()

    @staticmethod
    def _event_semantic_digest(events: list[dict[str, Any]]) -> str:
        canonical = [
            {
                "type": str(event.get("type") or ""),
                "kernel_step_id": str(event.get("kernel_step_id") or ""),
            }
            for event in list(events or [])
        ]
        canonical.sort(key=lambda item: (str(item.get("type") or ""), str(item.get("kernel_step_id") or "")))
        return hashlib.sha256(
            json.dumps(canonical, sort_keys=True, default=str).encode("utf-8"),
        ).hexdigest()

    @staticmethod
    def _load_archived_events(archive_path: str) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        if not archive_path:
            return events
        with gzip.open(archive_path, "rt", encoding="utf-8") as handle:
            for line in handle:
                stripped = str(line or "").strip()
                if not stripped:
                    continue
                try:
                    parsed = json.loads(stripped)
                    if isinstance(parsed, dict):
                        events.append(parsed)
                except json.JSONDecodeError:
                    continue
        return events

    def _compaction_losslessness_proof(
        self,
        *,
        pre_events: list[dict[str, Any]],
        post_events: list[dict[str, Any]],
        archive_path: str,
    ) -> dict[str, Any]:
        archived_events = self._load_archived_events(archive_path)
        reconstructed = sorted(
            [dict(event) for event in list(archived_events or [])] + [dict(event) for event in list(post_events or [])],
            key=lambda item: int(item.get("sequence") or 0),
        )
        pre_digest = self._event_stream_digest(list(pre_events or []))
        reconstructed_digest = self._event_stream_digest(reconstructed)
        sequence_equivalent = [
            int(item.get("sequence") or 0) for item in list(pre_events or [])
        ] == [
            int(item.get("sequence") or 0) for item in reconstructed
        ]
        return {
            "contract_version": "ledger-compaction-lossless-v1",
            "equivalent": bool(pre_digest == reconstructed_digest and sequence_equivalent),
            "pre_digest": pre_digest,
            "reconstructed_digest": reconstructed_digest,
            "archived_event_count": len(archived_events),
            "reconstructed_event_count": len(reconstructed),
            "sequence_equivalent": bool(sequence_equivalent),
        }

    @staticmethod
    def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
        raw = str(os.environ.get(name, str(default))).strip()
        value = int(raw) if raw.isdigit() else int(default)
        return max(int(minimum), value)

    @staticmethod
    def _env_bool(name: str, default: bool) -> bool:
        raw = str(os.environ.get(name, "1" if default else "0")).strip().lower()
        return raw in {"1", "true", "yes", "on"}

    def _ensure_compactor(self) -> EventCompactor:
        if self._ledger_compactor is None:
            max_events = self._env_int("DADBOT_LEDGER_MAX_EVENTS", 10000, minimum=100)
            max_age_seconds = float(self._env_int("DADBOT_LEDGER_MAX_AGE_SECONDS", 86400, minimum=60))
            min_snapshot_distance = self._env_int("DADBOT_LEDGER_MIN_SNAPSHOT_DISTANCE", 200, minimum=0)
            archive_dir = Path(
                str(os.environ.get("DADBOT_LEDGER_ARCHIVE_DIR", "runtime/archives")).strip()
                or "runtime/archives"
            )
            self._ledger_compactor = EventCompactor(
                policy=CompactionPolicy(
                    max_events=max_events,
                    max_age_seconds=max_age_seconds,
                    min_snapshot_distance=min_snapshot_distance,
                ),
                archive=ArchiveTier(archive_dir),
            )
        return self._ledger_compactor

    def _partition_summary(self, events: list[dict[str, Any]]) -> dict[str, Any]:
        by_session: dict[str, int] = {}
        for event in list(events or []):
            sid = str(event.get("session_id") or "").strip() or "unknown"
            by_session[sid] = int(by_session.get(sid, 0)) + 1
        top_sessions = sorted(by_session.items(), key=lambda item: (-int(item[1]), str(item[0])))[:8]
        return {
            "partition_count": len(by_session),
            "top_partitions": [{"session_id": sid, "event_count": int(count)} for sid, count in top_sessions],
        }

    def _maybe_compact_ledger(self) -> dict[str, Any]:
        hard_max = self._env_int("DADBOT_LEDGER_HARD_LIMIT_EVENTS", 50000, minimum=500)
        pre_events = list(self.ledger.read())
        event_count = len(pre_events)

        if event_count <= 0:
            return {"compacted": False, "reason": "empty", "event_count": 0, **self._partition_summary(pre_events)}

        force = bool(event_count >= hard_max)
        compactor = self._ensure_compactor()
        snapshot = {"head_sequence": event_count}
        report = dict(compactor.compact(ledger=self.ledger, snapshot=snapshot, force=force) or {})
        post_events = list(self.ledger.read())
        lossless_proof = {
            "contract_version": "ledger-compaction-lossless-v1",
            "equivalent": True,
            "pre_digest": self._event_stream_digest(pre_events),
            "reconstructed_digest": self._event_stream_digest(post_events),
            "archived_event_count": 0,
            "reconstructed_event_count": len(post_events),
            "sequence_equivalent": True,
        }
        archive_path = str(report.get("archive_path") or "")
        if bool(report.get("compacted", False)) and archive_path:
            lossless_proof = self._compaction_losslessness_proof(
                pre_events=pre_events,
                post_events=post_events,
                archive_path=archive_path,
            )
            if not bool(lossless_proof.get("equivalent", False)):
                raise RuntimeError("Ledger compaction losslessness invariant violated")
        report.setdefault("event_count", event_count)
        report.setdefault("forced", force)
        report["lossless_proof"] = dict(lossless_proof)
        report.update(self._partition_summary(post_events))
        self._last_compaction_report = report
        return report
