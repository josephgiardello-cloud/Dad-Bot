from __future__ import annotations

import asyncio
import json
import random
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from dadbot.core.control_plane import ExecutionControlPlane, SessionRegistry
from dadbot.core.durability import FileLockMutex
from dadbot.core.graph_context import TurnContext
from dadbot.core.nodes import SaveNode
from dadbot.memory.decay_policy import MemoryDecayPolicy

pytestmark = pytest.mark.unit


def test_inference_jitter_simulation_does_not_deadlock_serialized_session() -> None:
    rng = random.Random(1337)
    registry = SessionRegistry()

    async def _kernel_executor(_session, _job):
        await asyncio.sleep(rng.uniform(0.02, 0.08))
        return ("ok", False)

    control_plane = ExecutionControlPlane(registry=registry, kernel_executor=_kernel_executor)

    async def _run() -> list[tuple[str, bool]]:
        tasks = [
            control_plane.submit_turn(
                session_id="chaos-jitter",
                user_input=f"turn-{idx}",
                metadata={"request_id": f"req-{idx}", "trace_id": f"tr-{idx}"},
                timeout_seconds=8.0,
            )
            for idx in range(12)
        ]
        return await asyncio.gather(*tasks)

    results = asyncio.run(_run())
    assert len(results) == 12
    assert all(item == ("ok", False) for item in results)
    assert control_plane.execution_lease.owner_of("chaos-jitter") is None


class _SocketMutilationStorage:
    def __init__(self) -> None:
        self.begin_calls = 0
        self.apply_calls = 0
        self.commit_calls = 0
        self.rollback_calls = 0

    def begin_transaction(self, _ctx) -> None:
        self.begin_calls += 1

    def apply_mutations(self, _ctx) -> None:
        self.apply_calls += 1

    def finalize_turn(self, _ctx, _result):
        raise ConnectionError("socket mutilation: upstream model disconnected mid-stream")

    def commit_transaction(self, _ctx) -> None:
        self.commit_calls += 1

    def rollback_transaction(self, _ctx) -> None:
        self.rollback_calls += 1


def test_socket_mutilation_forces_save_node_rollback() -> None:
    storage = _SocketMutilationStorage()
    save = SaveNode(storage)
    ctx = TurnContext(user_input="socket test")
    ctx.state["candidate"] = "candidate-reply"

    with pytest.raises(ConnectionError, match="socket mutilation"):
        asyncio.run(save.run(ctx))

    assert storage.begin_calls == 1
    assert storage.apply_calls == 1
    assert storage.commit_calls == 0
    assert storage.rollback_calls == 1


def test_zombie_lock_recovers_on_bootstrap() -> None:
    with tempfile.TemporaryDirectory() as td:
        lock_path = Path(td) / "runtime_contract.lock"
        stale_record = {
            "pid": 999999,
            "token": "dead-token",
            "acquired_at": time.time() - 3600,
        }
        lock_path.write_text(json.dumps(stale_record), encoding="utf-8")

        lock = FileLockMutex(lock_path, stale_after_seconds=1.0)
        token = lock.acquire(timeout_seconds=2.0)
        assert token
        assert lock.is_held is True
        assert lock.release(token) is True
        assert lock_path.exists() is False


def test_memory_pressure_prefers_prune_over_crash() -> None:
    entries = [
        {
            "id": f"m-{idx}",
            "updated_at": "2000-01-01",
            "confidence": 0.1,
            "source_count": 1,
            "importance_score": 0.0,
        }
        for idx in range(3000)
    ]
    turn_context = SimpleNamespace(temporal=SimpleNamespace(epoch_seconds=1_800_000_000.0))
    policy = MemoryDecayPolicy(prune_threshold=0.70, weaken_threshold=0.90)

    result = policy.apply(entries, turn_context)
    assert len(result.pruned) >= 2500
    assert len(result.pruned) + len(result.weakened) + len(result.unchanged) == len(entries)


def _extract_fail_hard_reason(log_text: str) -> str:
    for line in str(log_text or "").splitlines():
        line = line.strip()
        if "FAIL_HARD" in line and "reason=" in line:
            return line.split("reason=", 1)[1].strip()
    return ""


def test_blind_debug_extracts_fail_hard_reason_from_logs_only() -> None:
    synthetic_logs = "\n".join(
        [
            "2026-05-02T10:11:00Z INFO turn_start trace=abc123",
            "2026-05-02T10:11:01Z ERROR FAIL_HARD stage=safety reason=policy_violation:block_harmful_output",
            "2026-05-02T10:11:01Z INFO turn_end status=failed",
        ]
    )
    reason = _extract_fail_hard_reason(synthetic_logs)
    assert reason == "policy_violation:block_harmful_output"
