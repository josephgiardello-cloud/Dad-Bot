from __future__ import annotations

import asyncio
import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from dadbot.core.invariance_contract import evaluation_contract_payload
from tests.stress.phase4_certification_gate import build_bot

ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "session_logs" / "execution_invariance_audit.json"


def _sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def _canonicalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _canonicalize(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_canonicalize(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_canonicalize(item) for item in value)
    return value


def _normalize_envelope_projection(result: dict[str, Any]) -> dict[str, str]:
    return {
        "determinism_manifest_hash": str(result.get("determinism_manifest_hash") or ""),
        "lock_hash": str(result.get("lock_hash") or ""),
    }


def _canonical_envelope_detail(runtime_mode: dict[str, Any], test_mode: dict[str, Any]) -> dict[str, Any]:
    runtime_projection = _normalize_envelope_projection(runtime_mode)
    test_projection = _normalize_envelope_projection(test_mode)
    fields = sorted(set(runtime_projection) | set(test_projection))
    pairs = [
        {
            "field": field,
            "runtime": runtime_projection.get(field, ""),
            "test": test_projection.get(field, ""),
            "matches": runtime_projection.get(field, "") == test_projection.get(field, ""),
        }
        for field in fields
    ]
    return {
        "fields": pairs,
        "diagnostic_hash": _sha256(_canonicalize(pairs)),
    }


def _evaluate_envelope_divergence(runtime_mode: dict[str, Any], test_mode: dict[str, Any]) -> dict[str, Any]:
    # Envelope fields: manifest hash, lock_hash — expected to vary across contexts.
    envelope_identical = (
        runtime_mode["determinism_manifest_hash"] == test_mode["determinism_manifest_hash"]
        and runtime_mode["lock_hash"] == test_mode["lock_hash"]
    )
    normalized_envelope_detail = _canonical_envelope_detail(runtime_mode, test_mode)

    # Behavioral fields cross-reference (informational only here).
    replay_hash_identical = runtime_mode["replay_hash"] == test_mode["replay_hash"]
    tool_trace_hash_identical = runtime_mode["tool_trace_hash"] == test_mode["tool_trace_hash"]

    return {
        # envelope_pass is informational; variation is expected and not a bug.
        "envelope_pass": envelope_identical,
        # behavioral fields included for cross-reference visibility.
        "behavioral_cross_reference": {
            "replay_hash_identical": replay_hash_identical,
            "tool_trace_hash_identical": tool_trace_hash_identical,
            "runtime_replay_hash": runtime_mode["replay_hash"],
            "test_replay_hash": test_mode["replay_hash"],
            "runtime_tool_trace_hash": runtime_mode["tool_trace_hash"],
            "test_tool_trace_hash": test_mode["tool_trace_hash"],
        },
        "envelope_detail": normalized_envelope_detail,
        "note": (
            "Envelope divergence is expected: _build_determinism_manifest() hashes "
            "env-var keys including PYTEST_CURRENT_TEST, so manifest/lock hashes "
            "legitimately differ between test and runtime contexts."
        ),
    }


def _build_audit_payload(
    *,
    trace_stability: dict[str, Any],
    orchestrator_determinism: dict[str, Any],
    envelope_divergence: dict[str, Any],
) -> dict[str, Any]:
    behavioral_invariance = (
        bool(trace_stability.get("passed"))
        and bool(orchestrator_determinism.get("passed"))
        and bool(envelope_divergence["behavioral_cross_reference"]["replay_hash_identical"])
        and bool(envelope_divergence["behavioral_cross_reference"]["tool_trace_hash_identical"])
    )
    envelope_invariance = bool(envelope_divergence.get("envelope_pass"))
    return {
        "audit": "execution_invariance",
        "evaluation_contract": _canonicalize(evaluation_contract_payload()),
        "behavioral_invariance": {
            "passed": behavioral_invariance,
            "description": "CORE - does the system behave identically? replay_hash + tool_trace_hash stability.",
            "checks": {
                "trace_fingerprint_stability_under_repeated_runs": trace_stability,
                "orchestrator_determinism_under_identical_inputs": orchestrator_determinism,
            },
        },
        "envelope_invariance": {
            "passed": envelope_invariance,
            "description": "SECONDARY - is the execution context identical? env vars, manifest metadata. Informational only.",
            "checks": {
                "instrumentation_envelope_divergence": envelope_divergence,
            },
        },
        "overall_pass": behavioral_invariance,
    }


@contextmanager
def _temporary_env_var(key: str, value: str | None):
    old = os.environ.get(key)
    try:
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
        yield
    finally:
        if old is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = old


def _run_one_orchestrator_turn(*, work_dir: Path, user_input: str, instrumentation_mode: str) -> dict[str, Any]:
    mode_value = None if instrumentation_mode == "runtime" else "pytest::simulated"
    with _temporary_env_var("PYTEST_CURRENT_TEST", mode_value):
        bot = build_bot(work_dir)
        try:
            metadata = {
                "audit_mode": True,
                "trace_id": "exec-invariance-trace",
                "correlation_id": "exec-invariance-corr",
                "request_id": "exec-invariance-req",
            }
            _ = asyncio.run(
                bot.turn_orchestrator.control_plane.submit_turn(
                    session_id="exec-invariance",
                    user_input=user_input,
                    metadata=metadata,
                )
            )

            session = bot.turn_orchestrator.session_registry.get_or_create("exec-invariance")
            state = dict(session.get("state") or {})
            last_det = dict(
                state.get("last_determinism") or {}
            )
            manifest = dict(
                state.get("last_determinism_manifest") or {}
            )

            return {
                "replay_hash": str(bot.turn_orchestrator.control_plane.ledger.replay_hash() or ""),
                "tool_trace_hash": str(last_det.get("tool_trace_hash") or ""),
                "lock_hash": str(last_det.get("lock_hash") or ""),
                "lock_hash_with_tools": str(last_det.get("lock_hash_with_tools") or ""),
                "determinism_manifest_hash": _sha256(manifest),
                "mode": instrumentation_mode,
            }
        finally:
            bot.shutdown()


def _trace_fingerprint_stability() -> dict[str, Any]:
    prompts = [
        "fact:job=stressed",
        "fact:budget=tight",
        "fact:sleep=poor",
        "fact:job=focused",
        "fact:budget=stable",
    ]

    def _collect_trace_signature(root: Path) -> dict[str, Any]:
        bot = build_bot(root)
        try:
            per_turn = []
            for idx, prompt in enumerate(prompts, start=1):
                metadata = {
                    "audit_mode": True,
                    "trace_id": f"trace-stability-{idx}",
                    "correlation_id": f"trace-stability-corr-{idx}",
                    "request_id": f"trace-stability-req-{idx}",
                }
                _ = asyncio.run(
                    bot.turn_orchestrator.control_plane.submit_turn(
                        session_id="trace-stability",
                        user_input=prompt,
                        metadata=metadata,
                    )
                )

                session_state = dict(
                    (bot.turn_orchestrator.session_registry.get_or_create("trace-stability") or {}).get("state")
                    or {}
                )
                det = dict(session_state.get("last_determinism") or {})
                per_turn.append(
                    {
                        "lock_hash": str(det.get("lock_hash") or ""),
                        "tool_trace_hash": str(det.get("tool_trace_hash") or ""),
                        "lock_hash_with_tools": str(det.get("lock_hash_with_tools") or ""),
                    }
                )

            return {
                "per_turn": per_turn,
                "replay_hash": str(bot.turn_orchestrator.control_plane.ledger.replay_hash() or ""),
            }
        finally:
            bot.shutdown()

    with TemporaryDirectory() as left_tmp, TemporaryDirectory() as right_tmp:
        run1 = _collect_trace_signature(Path(left_tmp))
        run2 = _collect_trace_signature(Path(right_tmp))

    trace_hash_1 = _sha256(run1)
    trace_hash_2 = _sha256(run2)

    return {
        "passed": trace_hash_1 == trace_hash_2,
        "run1_trace_hash": trace_hash_1,
        "run2_trace_hash": trace_hash_2,
        "run1": run1,
        "run2": run2,
    }


def _orchestrator_determinism_identical_inputs() -> dict[str, Any]:
    with TemporaryDirectory() as tmp_a, TemporaryDirectory() as tmp_b:
        left = _run_one_orchestrator_turn(
            work_dir=Path(tmp_a),
            user_input="Walk me through the same decision in a deterministic way.",
            instrumentation_mode="runtime",
        )
        right = _run_one_orchestrator_turn(
            work_dir=Path(tmp_b),
            user_input="Walk me through the same decision in a deterministic way.",
            instrumentation_mode="runtime",
        )

    passed = (
        left["replay_hash"] == right["replay_hash"]
        and left["tool_trace_hash"] == right["tool_trace_hash"]
        and left["lock_hash_with_tools"] == right["lock_hash_with_tools"]
    )

    return {
        "passed": passed,
        "left": left,
        "right": right,
    }


def _instrumentation_envelope_divergence() -> dict[str, Any]:
    """
    SECONDARY / envelope gate.

    Compares manifest-level metadata (env-var keys, instrumentation flags) between
    runtime and test modes.  replay_hash and tool_trace_hash are the *behavioral*
    signals and are also reported here for cross-reference, but they are NOT the
    pass/fail criterion for this check — they belong to the behavioral gate.
    """
    with TemporaryDirectory() as tmp_runtime, TemporaryDirectory() as tmp_test:
        runtime_mode = _run_one_orchestrator_turn(
            work_dir=Path(tmp_runtime),
            user_input="Audit parity between runtime and test instrumentation.",
            instrumentation_mode="runtime",
        )
        test_mode = _run_one_orchestrator_turn(
            work_dir=Path(tmp_test),
            user_input="Audit parity between runtime and test instrumentation.",
            instrumentation_mode="test",
        )
    return _evaluate_envelope_divergence(runtime_mode, test_mode)


def main() -> int:
    trace_stability = _trace_fingerprint_stability()
    orchestrator_determinism = _orchestrator_determinism_identical_inputs()
    envelope_divergence = _instrumentation_envelope_divergence()
    payload = _build_audit_payload(
        trace_stability=trace_stability,
        orchestrator_determinism=orchestrator_determinism,
        envelope_divergence=envelope_divergence,
    )
    behavioral_invariance = bool(payload["behavioral_invariance"].get("passed"))
    envelope_invariance = bool(payload["envelope_invariance"].get("passed"))
    overall_pass = bool(payload.get("overall_pass"))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True), encoding="utf-8")

    print(f"WROTE: {OUT_PATH}")
    print(f"BEHAVIORAL_INVARIANCE={behavioral_invariance}  [CORE — correctness gate]")
    print(f"ENVELOPE_INVARIANCE={envelope_invariance}  [SECONDARY — informational]")
    print(f"OVERALL_PASS={overall_pass}")
    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
