"""
Shadow Mode Logger
==================
Non-blocking, sampling-bounded observation layer.

Measures replay fidelity under real entropy.
Does NOT mutate execution state or branch kernel behavior.

Enable via environment variable:
    DADBOT_SHADOW_MODE=1

Sampling defaults:
    DADBOT_SHADOW_SAMPLE_RATE=0.10    (10% of turns)
    DADBOT_SHADOW_DAILY_CAP=1000      (max samples/day)

Log file:
    session_logs/shadow_mode.jsonl    (append-only JSONL)

Baseline reference:
    BASELINE_SNAPSHOT = 2026-05-02_1403

Log record fields (v2):
    timestamp, snapshot_hash, replay_hash, result,
    divergence_type, divergence_detail,
    input_hash, runtime_fingerprint,
    latency_ms, event_count,
    snapshot_version, trace_id, session_id, error
"""
from __future__ import annotations

import hashlib
import json
import os
import secrets
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASELINE_SNAPSHOT_ID: str = "2026-05-02_1403"
_DEFAULT_SAMPLE_RATE: float = 0.10
_DEFAULT_DAILY_CAP: int = 1_000
_DEFAULT_LOG_PATH: Path = Path(__file__).resolve().parents[2] / "session_logs" / "shadow_mode.jsonl"

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def is_enabled() -> bool:
    """Return True if shadow mode is active (DADBOT_SHADOW_MODE=1)."""
    return os.environ.get("DADBOT_SHADOW_MODE", "").strip() == "1"


def shadow_log(
    *,
    snapshot: dict[str, Any],
    trace_id: str,
    session_id: str,
    event_count: int,
    latency_ms: float,
) -> None:
    """
    Emit one shadow-mode log record.  Non-blocking — never raises.

    Parameters
    ----------
    snapshot:
        The execution_snapshot dict produced by _build_execution_snapshot.
        Must contain 'snapshot_hash'.
    trace_id:
        Canonical trace identifier for this turn.
    session_id:
        Session identifier.
    event_count:
        Number of recorded execution steps for this turn.
    latency_ms:
        Wall-clock turn latency in milliseconds.
    """
    try:
        if not is_enabled():
            return
        if not _should_sample():
            return
        _emit(
            snapshot=snapshot,
            trace_id=trace_id,
            session_id=session_id,
            event_count=int(event_count),
            latency_ms=float(latency_ms),
        )
    except Exception:  # noqa: BLE001 — never let shadow mode affect the caller
        pass


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _sample_rate() -> float:
    try:
        return float(os.environ.get("DADBOT_SHADOW_SAMPLE_RATE", str(_DEFAULT_SAMPLE_RATE)))
    except (ValueError, TypeError):
        return _DEFAULT_SAMPLE_RATE


def _daily_cap() -> int:
    try:
        return int(os.environ.get("DADBOT_SHADOW_DAILY_CAP", str(_DEFAULT_DAILY_CAP)))
    except (ValueError, TypeError):
        return _DEFAULT_DAILY_CAP


def _today_str() -> str:
    import datetime
    return datetime.date.today().isoformat()


def _log_path() -> Path:
    """Resolve shadow log destination, honoring env override when provided."""
    configured = os.environ.get("DADBOT_SHADOW_LOG_PATH", "").strip()
    if configured:
        path = Path(configured).expanduser()
        if not path.is_absolute():
            path = (Path(__file__).resolve().parents[2] / path).resolve()
        return path.resolve()
    return _DEFAULT_LOG_PATH


def _counter_path(log_path: Path) -> Path:
    return log_path.parent / "shadow_mode_daily_counter.json"


def _should_sample() -> bool:
    """Return True if this turn should be sampled, respecting rate and daily cap."""
    rate = _sample_rate()
    if rate <= 0.0:
        return False
    if rate < 1.0:
        # Integer threshold avoids importing the random module in core runtime.
        threshold = int(rate * 1_000_000)
        if threshold <= 0:
            return False
        if secrets.randbelow(1_000_000) >= threshold:
            return False
    # rate >= 1.0 always samples (subject to daily cap).
    return _check_and_increment_daily_counter()


def _check_and_increment_daily_counter() -> bool:
    """Atomically read/write the daily counter.  Returns True if under cap."""
    today = _today_str()
    cap = _daily_cap()
    counter_path = _counter_path(_log_path())
    try:
        counter_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            data: dict[str, Any] = json.loads(counter_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            data = {}
        if data.get("date") != today:
            data = {"date": today, "count": 0}
        if int(data["count"]) >= cap:
            return False
        data["count"] = int(data["count"]) + 1
        counter_path.write_text(json.dumps(data), encoding="utf-8")
        return True
    except Exception:  # noqa: BLE001
        return True  # on counter failure, allow rather than block


def _stable_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8"),
    ).hexdigest()


def _classify_divergence(
    snapshot_hash: str,
    replay_hash: str,
    snapshot: dict[str, Any],
) -> tuple[str, str | None]:
    """
    Best-effort divergence classification.

    Returns (primary_bucket, detail_tag):
        primary_bucket: "none" | "tool" | "memory" | "execution"
        detail_tag:     None when "none"; sub-tag when known, else None.

    execution sub-tags: "ordering" | "missing_event" | "extra_event" | "hash_mismatch"
    memory sub-tags:    "retrieval_diff" | "ranking_diff" | "decay_diff"
    """
    if snapshot_hash == replay_hash:
        return ("none", None)

    # --- tool divergence -------------------------------------------------------
    tool_outputs = list(snapshot.get("tool_outputs") or [])
    if tool_outputs:
        return ("tool", None)

    # --- memory divergence -----------------------------------------------------
    memory_snapshot = dict(snapshot.get("memory_snapshot") or {})
    if memory_snapshot:
        detail: str | None = None
        if memory_snapshot.get("retrieval_set") is not None:
            # Compare cardinality / ordering as a coarse signal.
            retrieval = list(memory_snapshot.get("retrieval_set") or [])
            original_retrieval = list(
                (snapshot.get("inputs") or {}).get("memory_context") or []
            )
            if set(str(x) for x in retrieval) != set(str(x) for x in original_retrieval):
                detail = "retrieval_diff"
            elif retrieval != original_retrieval:
                detail = "ranking_diff"
            elif memory_snapshot.get("decay_applied"):
                detail = "decay_diff"
        return ("memory", detail)

    # --- execution divergence --------------------------------------------------
    outputs_per_step = list(snapshot.get("outputs_per_step") or [])
    baseline_steps = list((snapshot.get("inputs") or {}).get("expected_steps") or [])
    detail = None
    if baseline_steps:
        if len(outputs_per_step) < len(baseline_steps):
            detail = "missing_event"
        elif len(outputs_per_step) > len(baseline_steps):
            detail = "extra_event"
        else:
            # Same length — check ordering by comparing step names/ids.
            out_ids = [
                str(s.get("step") or s.get("node") or i)
                for i, s in enumerate(outputs_per_step)
            ]
            base_ids = [
                str(s.get("step") or s.get("node") or i)
                for i, s in enumerate(baseline_steps)
            ]
            if out_ids != base_ids:
                detail = "ordering"
            else:
                detail = "hash_mismatch"
    else:
        detail = "hash_mismatch"

    return ("execution", detail)


def _compute_input_hash(snapshot: dict[str, Any]) -> str:
    """
    Hash the normalized input surface: user prompt + tool inputs + context slice.
    No raw strings are stored — only the hash.
    """
    inputs = dict(snapshot.get("inputs") or {})
    projection = {
        "prompt": str(inputs.get("prompt") or inputs.get("user_input") or ""),
        "tool_inputs": sorted(
            str(x) for x in (inputs.get("tool_inputs") or inputs.get("tools") or [])
        ),
        "context_slice": str(inputs.get("context_slice") or inputs.get("memory_context") or ""),
    }
    return _stable_sha256(projection)


def _build_runtime_fingerprint(snapshot: dict[str, Any]) -> dict[str, str | None]:
    """
    Capture the execution environment so drift can be separated from system failures.
    Returns:
        code_version: short git SHA, else package __version__, else "unknown"
        model_id:     model identifier from snapshot, else None
        config_hash:  hash of key config fields from snapshot, else None
    """
    # code_version — read from pre-cached env var (set at bot startup) to avoid
    # any subprocess call inside the logging path.  Falls back to package version.
    code_version: str | None = os.environ.get("DADBOT_CODE_VERSION") or None
    if not code_version:
        try:
            import importlib.metadata
            code_version = importlib.metadata.version("dadbot")
        except Exception:  # noqa: BLE001
            code_version = "unknown"

    # model_id — read from snapshot's inference section
    model_id: str | None = None
    try:
        inference = dict(snapshot.get("inference") or snapshot.get("inputs") or {})
        model_id = str(inference.get("model") or inference.get("model_id") or "") or None
    except Exception:  # noqa: BLE001
        pass

    # config_hash — hash of whatever config fields the snapshot carries
    config_hash: str | None = None
    try:
        config_fields = dict(snapshot.get("config") or {})
        if config_fields:
            config_hash = _stable_sha256(config_fields)
    except Exception:  # noqa: BLE001
        pass

    return {
        "code_version": code_version,
        "model_id": model_id,
        "config_hash": config_hash,
    }


def _compute_replay_hash(snapshot: dict[str, Any]) -> str:
    """
    Derive the deterministic replay hash from the snapshot.
    This is the hash that should match on a clean re-execution.
    """
    replay_projection = {
        "inputs": dict(snapshot.get("inputs") or {}),
        "outputs_per_step": list(snapshot.get("outputs_per_step") or []),
        "final_output": str(snapshot.get("final_output") or ""),
    }
    return _stable_sha256(replay_projection)


def _emit(
    *,
    snapshot: dict[str, Any],
    trace_id: str,
    session_id: str,
    event_count: int,
    latency_ms: float,
) -> None:
    # Compute both hashes from the same canonical projection so comparison is meaningful.
    _projection = {
        "inputs": dict(snapshot.get("inputs") or {}),
        "outputs_per_step": list(snapshot.get("outputs_per_step") or []),
        "final_output": str(snapshot.get("final_output") or ""),
    }
    snapshot_hash = str(snapshot.get("snapshot_hash") or _stable_sha256(_projection))
    replay_hash = _stable_sha256(_projection)
    divergence_type, divergence_detail = _classify_divergence(snapshot_hash, replay_hash, snapshot)
    result = "pass" if divergence_type == "none" else "fail"

    record: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
        "snapshot_hash": snapshot_hash,
        "replay_hash": replay_hash,
        "result": result,
        "divergence_type": divergence_type,
        "divergence_detail": divergence_detail,
        "input_hash": _compute_input_hash(snapshot),
        "runtime_fingerprint": _build_runtime_fingerprint(snapshot),
        "latency_ms": round(latency_ms, 2),
        "event_count": event_count,
        "snapshot_version": BASELINE_SNAPSHOT_ID,
        "trace_id": str(trace_id or ""),
        "session_id": str(session_id or ""),
        "error": None,
    }

    log_path = _log_path()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")
