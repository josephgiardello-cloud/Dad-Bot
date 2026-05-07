"""DadBotHealthMixin — health/UX state and checkpoint/replay for DadBot.

Extracted from the DadBot god-class. Owns:
- runtime_health_snapshot and its cached variant
- turn_health_state, turn_fidelity_state, turn_ux_feedback
  (all read from cached state stamped by TurnUxProjectionGateway)
- Checkpoint/replay delegates (conversation_persistence + OfflineReplayValidator)
"""

from __future__ import annotations

import time
from typing import Any


class DadBotHealthMixin:
    """Health/UX state and checkpoint/replay delegates for the DadBot facade."""

    # ------------------------------------------------------------------
    # Runtime health snapshot
    # ------------------------------------------------------------------

    def runtime_health_snapshot(
        self,
        *,
        log_warnings: bool = True,
        persist: bool = True,
    ) -> dict:
        health_manager = getattr(self, "health_manager", None)
        snapshot_fn = getattr(health_manager, "runtime_health_snapshot", None)
        if callable(snapshot_fn):
            return snapshot_fn(log_warnings=log_warnings, persist=persist)
        return {"status": "unknown"}

    def current_runtime_health_snapshot(
        self,
        *,
        force: bool = False,
        log_warnings: bool = False,
        persist: bool = False,
        max_age_seconds=None,
    ) -> dict:
        """Cached health snapshot; explicit on DadBot so tests can patch it."""
        now = time.monotonic()
        max_age_seconds = (
            max(0, int(max_age_seconds))
            if max_age_seconds is not None
            else max(30, int(self._health_snapshot_interval_seconds or 300))
        )
        cached = self._cached_runtime_health_snapshot
        last = self._last_runtime_health_snapshot_monotonic
        if not force and isinstance(cached, dict) and (now - last) <= max_age_seconds:
            return dict(cached)
        snapshot = getattr(
            self,
            "runtime_health_snapshot",
            lambda **_kw: {"status": "unknown"},
        )(log_warnings=log_warnings, persist=persist)
        self._cached_runtime_health_snapshot = dict(snapshot)
        self._last_runtime_health_snapshot_monotonic = now
        return dict(snapshot)

    # ------------------------------------------------------------------
    # Turn health / fidelity / UX feedback
    # These are read-only views of cached state stamped by
    # TurnUxProjectionGateway after each graph execution.
    # DadBot does NOT access ctx.state or fidelity directly.
    # ------------------------------------------------------------------

    def turn_health_state(self) -> dict[str, Any]:
        """Get cached turn health state from last execution."""
        payload = dict(getattr(self, "_last_turn_health_state", {}) or {})
        if payload:
            return payload
        return {
            "status": "OK",
            "latency_ms": 0.0,
            "memory_ops_time": 0.0,
            "graph_sync_time": 0.0,
            "inference_time": 0.0,
            "fallback_used": False,
        }

    def turn_fidelity_state(self) -> dict[str, Any]:
        """Get cached turn fidelity state from last execution."""
        payload = dict(getattr(self, "_last_turn_health_state", {}) or {})
        fidelity = dict(payload.get("fidelity") or {}) if isinstance(payload, dict) else {}
        if fidelity:
            return {
                "temporal": bool(fidelity.get("temporal", False)),
                "inference": bool(fidelity.get("inference", False)),
                "reflection": bool(fidelity.get("reflection", False)),
                "save": bool(fidelity.get("save", False)),
                "full_pipeline": bool(fidelity.get("full_pipeline", False)),
            }
        return {
            "temporal": False,
            "inference": False,
            "reflection": False,
            "save": False,
            "full_pipeline": False,
        }

    def turn_ux_feedback(self) -> dict[str, Any]:
        """Get cached UX feedback from last execution."""
        payload = dict(getattr(self, "_last_turn_ux_feedback", {}) or {})
        if payload:
            return payload
        return {
            "dad_is_thinking": False,
            "message": "",
            "checking_memory": False,
            "memory_message": "",
            "mood_hint": str(self.last_saved_mood() or "neutral"),
            "status": "OK",
        }

    # ------------------------------------------------------------------
    # Checkpoint and replay delegates
    # ------------------------------------------------------------------

    def load_latest_graph_checkpoint(self, trace_id: str = "") -> dict[str, Any] | None:
        return self.conversation_persistence.load_latest_graph_checkpoint(
            trace_id=trace_id,
        )

    def resume_turn_from_checkpoint(self, trace_id: str = "") -> dict[str, Any] | None:
        return self.conversation_persistence.resume_graph_checkpoint(trace_id=trace_id)

    def list_turn_events(self, trace_id: str, limit: int = 0) -> list[dict[str, Any]]:
        return self.conversation_persistence.list_turn_events(
            trace_id=trace_id,
            limit=limit,
        )

    def replay_turn_events(self, trace_id: str) -> dict[str, Any]:
        return self.conversation_persistence.replay_turn_events(trace_id=trace_id)

    def list_policy_trace_events(
        self,
        *,
        trace_id: str = "",
        limit: int = 0,
    ) -> list[dict[str, Any]]:
        return self.conversation_persistence.list_policy_trace_events(
            trace_id=trace_id,
            limit=limit,
        )

    def summarize_policy_trace_events(
        self,
        *,
        trace_id: str = "",
        limit: int = 0,
    ) -> dict[str, Any]:
        return self.conversation_persistence.summarize_policy_trace_events(
            trace_id=trace_id,
            limit=limit,
        )

    def validate_replay_determinism(
        self,
        trace_id: str,
        expected_lock_hash: str = "",
    ) -> dict[str, Any]:
        """Validate replay determinism using OfflineReplayValidator.

        Routes through: OfflineReplayValidator.validate_full()
        """
        from dadbot.core.offline_replay_validator import OfflineReplayValidator

        replay = self.conversation_persistence.replay_turn_events(trace_id=trace_id)
        determinism = dict(replay.get("determinism") or {})

        events = list(replay.get("events") or [])
        contract = dict(determinism.get("contract") or {})
        identity = dict(determinism.get("execution_identity") or {})

        validator = OfflineReplayValidator()
        report = validator.validate_full(
            contract=contract,
            events=events,
            identity=identity,
            trace_id=str(trace_id or "").strip(),
        )

        observed_hash = str(determinism.get("lock_hash") or "").strip()
        expected_hash = str(expected_lock_hash or "").strip()
        matches_expected = True
        if expected_hash:
            matches_expected = observed_hash == expected_hash

        return {
            "trace_id": str(trace_id or "").strip(),
            "replay_valid": report.passed,
            "replay_verdict": report.verdict,
            "replay_violations": report.violations,
            "consistent": bool(determinism.get("consistent", True)),
            "observed_lock_hash": observed_hash,
            "expected_lock_hash": expected_hash,
            "matches_expected": matches_expected,
            "lock_hashes": list(determinism.get("lock_hashes") or []),
            "execution_identity": identity,
            "execution_fingerprint": str(
                determinism.get("execution_fingerprint") or "",
            ),
        }
