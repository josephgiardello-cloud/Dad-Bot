"""Phase 4 Global Invariant Compliance Gate.

Covers four invariant categories:

  A. Determinism  — stage execute() methods must not call datetime.now() or
                    random sampling directly; time access is owned by
                    TurnTemporalAxis / VirtualClock.

  B. Mutation safety — every MutationIntent must carry payload_hash and
                       sequence_id; MutationGuard enforces SaveNode boundary.

  C. Graph integrity — TemporalNode precedes mutation-capable stages; SaveNode
                       always appears in the canonical pipeline.

  D. Persistence contract — PersistenceServiceContract exposes the minimum
                            surface; strict mode raises on violation rather than
                            silently degrading.

Run the whole suite:
    pytest tests/phase4 -vv

Strict mode (treats warnings as hard failures):
    PHASE4_STRICT=1 pytest tests/phase4 -vv
"""
from __future__ import annotations

import ast
import importlib
import os
import sys
from pathlib import Path
from textwrap import dedent
from types import SimpleNamespace
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Repo root and scope constants
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
DADBOT_SRC = ROOT / "dadbot"
STRICT = os.environ.get("PHASE4_STRICT", "").strip() not in ("", "0", "false", "no")

# Files where datetime.now() / time.time() are ALLOWED (temporal layer and
# infra-timing legitimately need wall-clock access).
_TEMPORAL_LAYER_ALLOW = frozenset(
    {
        "dadbot/core/graph.py",           # TurnTemporalAxis.from_now(), MutationTransactionRecord
        "dadbot/core/execution_ledger.py",# ledger timestamp stamps
        "dadbot/core/authorization.py",   # token expiry math
        "dadbot/state.py",                # state snapshot updated_at
        "dadbot/managers/conversation_persistence.py",  # log timestamps
        "dadbot/core/compat_mixin.py",    # compat helper
        "dadbot/services/turn_service.py",# pipeline timestamp helper
        "dadbot/background.py",           # background task timing
        "dadbot/agentic.py",              # email/calendar helpers (UI layer)
        "dadbot/infrastructure/storage.py",  # file naming
        "dadbot/managers/advice_audit.py",   # audit timestamps
        "dadbot/memory/graph_manager.py",    # memory graph wall_time
        "dadbot/core/control_plane.py",      # control plane ticket timestamps
    }
)


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


# ---------------------------------------------------------------------------
# A. Determinism invariants
# ---------------------------------------------------------------------------


class _ExecuteBodyVisitor(ast.NodeVisitor):
    """Collect direct datetime.now() / random calls inside execute() bodies."""

    BANNED_ATTRS = frozenset({"now"})  # datetime.now
    BANNED_FUNCS = frozenset({"random", "randint", "choice", "shuffle", "sample", "uniform"})

    def __init__(self) -> None:
        self.violations: list[tuple[int, str]] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        if node.name == "execute":
            self._scan_execute_body(node)
        # Do NOT recurse into nested functions/classes so inner execute defs
        # don't accidentally suppress the outer scan.
        self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]

    def _scan_execute_body(self, func_node: ast.FunctionDef) -> None:  # noqa: N802
        for child in ast.walk(func_node):
            # datetime.now() → Attribute call on 'datetime' name
            if isinstance(child, ast.Call):
                func = child.func
                if (
                    isinstance(func, ast.Attribute)
                    and func.attr in self.BANNED_ATTRS
                    and isinstance(func.value, ast.Name)
                    and func.value.id == "datetime"
                ):
                    self.violations.append(
                        (getattr(child, "lineno", 0), "datetime.now() in execute()")
                    )
                # random.xxx() calls
                if (
                    isinstance(func, ast.Attribute)
                    and func.attr in self.BANNED_FUNCS
                    and isinstance(func.value, ast.Name)
                    and func.value.id == "random"
                ):
                    self.violations.append(
                        (getattr(child, "lineno", 0), f"random.{func.attr}() in execute()")
                    )


def test_no_datetime_now_in_stage_execute_methods():
    """Stage node execute() methods must not call datetime.now() directly.

    All temporal access must route through TurnTemporalAxis or VirtualClock.
    Files in the temporal layer allow-list are excluded.
    """
    violations: list[str] = []
    for py_file in DADBOT_SRC.rglob("*.py"):
        rel = _rel(py_file)
        if any(rel.endswith(a.lstrip("./")) for a in _TEMPORAL_LAYER_ALLOW):
            continue
        try:
            tree = ast.parse(py_file.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        visitor = _ExecuteBodyVisitor()
        visitor.visit(tree)
        for line, msg in visitor.violations:
            violations.append(f"{rel}:{line} — {msg}")

    assert not violations, (
        "Determinism violation: stage execute() methods must not call datetime.now().\n"
        + "\n".join(f"  {v}" for v in violations)
    )


def test_no_unseeded_random_imports_in_core():
    """dadbot/core and dadbot/services must not import the 'random' module.

    Non-deterministic sampling in the execution path breaks replay equivalence.
    Use DeterminismBoundary.seal() with an explicit seed instead.
    """
    # Known exceptions: files that legitimately use random for non-execution-path
    # purposes (e.g., tool sandbox jitter, test harness noise generation).
    # Each entry here is a tracking acknowledgement — NOT a blanket permission.
    EXEMPT_PATTERNS = frozenset(
        {
            "dadbot/core/external_tool_runtime.py",  # sandbox jitter, not in turn execution path
        }
    )

    violations: list[str] = []
    for directory in ("core", "services"):
        for py_file in (DADBOT_SRC / directory).rglob("*.py"):
            rel = _rel(py_file)
            try:
                tree = ast.parse(py_file.read_text(encoding="utf-8", errors="replace"))
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    names = (
                        [alias.name for alias in node.names]
                        if isinstance(node, ast.Import)
                        else ([str(node.module or "")] if node.module else [])
                    )
                    if any(n == "random" or n.startswith("random.") for n in names):
                        if rel not in EXEMPT_PATTERNS:
                            violations.append(
                                f"{rel}:{getattr(node, 'lineno', 0)} — 'import random'"
                            )

    assert not violations, (
        "Determinism violation: core/services must not import 'random'.\n"
        + "\n".join(f"  {v}" for v in violations)
    )


def test_virtual_clock_and_temporal_axis_exist():
    """VirtualClock and TurnTemporalAxis are the approved temporal entry points."""
    graph_mod = importlib.import_module("dadbot.core.graph")
    assert hasattr(graph_mod, "VirtualClock"), "VirtualClock missing from dadbot.core.graph"
    assert hasattr(graph_mod, "TurnTemporalAxis"), "TurnTemporalAxis missing from dadbot.core.graph"
    vc = graph_mod.VirtualClock(base_epoch=1_700_000_000.0, step_size_seconds=1.0)
    t1 = vc.tick()
    t2 = vc.tick()
    assert t2 > t1, "VirtualClock.tick() must be monotonically increasing"


# ---------------------------------------------------------------------------
# B. Mutation safety invariants
# ---------------------------------------------------------------------------


def test_mutation_intent_has_required_fields():
    """MutationIntent must carry payload_hash and sequence_id after construction."""
    graph_mod = importlib.import_module("dadbot.core.graph")
    intent = graph_mod.MutationIntent(
        type="ledger",
        payload={
            "op": "append_history",
            "temporal": {"wall_time": "2024-01-01T00:00:00", "wall_date": "2024-01-01"},
        },
        requires_temporal=True,
        source="phase4_compliance_test",
    )
    assert hasattr(intent, "payload_hash"), "MutationIntent must have payload_hash"
    assert hasattr(intent, "sequence_id"), "MutationIntent must have sequence_id"
    assert isinstance(intent.payload_hash, str) and len(intent.payload_hash) > 0, (
        "payload_hash must be a non-empty string"
    )


def test_mutation_guard_blocks_queue_outside_save_node():
    """MutationGuard must raise RuntimeError when mutations are queued under guard."""
    graph_mod = importlib.import_module("dadbot.core.graph")
    ctx = graph_mod.TurnContext(user_input="guard test")
    intent = graph_mod.MutationIntent(
        type="ledger",
        payload={
            "op": "append_history",
            "temporal": {"wall_time": "2024-01-01T00:00:00", "wall_date": "2024-01-01"},
        },
        requires_temporal=True,
        source="guard_test",
    )
    with graph_mod.MutationGuard(ctx.mutation_queue):
        with pytest.raises(RuntimeError, match="MutationGuard violation"):
            ctx.mutation_queue.queue(intent)


def test_mutation_queue_drain_is_transactional():
    """drain(transactional=True) must roll back applied mutations on failure."""
    graph_mod = importlib.import_module("dadbot.core.graph")
    ctx = graph_mod.TurnContext(user_input="tx drain test")

    applied: list[str] = []
    rolled_back: list[str] = []

    def _executor(intent: Any) -> Any:
        tag = str(intent.source or "")
        if tag == "fail":
            raise RuntimeError("intentional drain failure")
        applied.append(tag)

        def _compensate() -> None:
            applied.remove(tag)
            rolled_back.append(tag)

        return _compensate

    for label in ("ok1", "ok2", "fail"):
        ctx.mutation_queue.queue(
            graph_mod.MutationIntent(
                type="ledger",
                payload={
                    "op": "append_history",
                    "temporal": {"wall_time": "2024-01-01T00:00:00", "wall_date": "2024-01-01"},
                },
                requires_temporal=True,
                source=label,
            )
        )

    with pytest.raises(graph_mod.FatalTurnError):
        ctx.mutation_queue.drain(_executor, transactional=True)

    # Rollback runs in reverse application order (LIFO), so ok2 is rolled back before ok1.
    assert set(rolled_back) == {"ok1", "ok2"}, (
        "Transactional drain must roll back all prior committed mutations on failure"
    )
    assert len(rolled_back) == 2, "All two applied mutations must be rolled back"
    assert applied == [], "No mutations should remain applied after rollback"


def test_mutation_intent_bad_kind_raises():
    """MutationIntent must reject unknown mutation kinds at construction time."""
    graph_mod = importlib.import_module("dadbot.core.graph")
    with pytest.raises(RuntimeError):
        graph_mod.MutationIntent(
            type="not_a_real_kind",
            payload={},
            requires_temporal=False,
        )


# ---------------------------------------------------------------------------
# C. Graph integrity invariants
# ---------------------------------------------------------------------------


def test_default_turn_graph_temporal_is_first_node():
    """TemporalNode must be the first node in the default TurnGraph pipeline."""
    graph_mod = importlib.import_module("dadbot.core.graph")
    graph = graph_mod.TurnGraph()
    names = [
        str(getattr(n, "name", type(n).__name__) or type(n).__name__)
        for n in graph.nodes
    ]
    assert names, "TurnGraph.nodes must not be empty"
    assert names[0] == "temporal", (
        f"TurnGraph.nodes[0] must be 'temporal', got {names[0]!r}"
    )


def test_default_turn_graph_save_node_present():
    """SaveNode must appear in the default TurnGraph pipeline."""
    graph_mod = importlib.import_module("dadbot.core.graph")
    graph = graph_mod.TurnGraph()
    names = [
        str(getattr(n, "name", type(n).__name__) or type(n).__name__)
        for n in graph.nodes
    ]
    assert "save" in names, f"TurnGraph default pipeline missing 'save'; got {names!r}"


def test_default_turn_graph_temporal_before_mutation_stages():
    """'temporal' must precede 'inference', 'save', and 'reflection'."""
    graph_mod = importlib.import_module("dadbot.core.graph")
    graph = graph_mod.TurnGraph()
    names = [
        str(getattr(n, "name", type(n).__name__) or type(n).__name__)
        for n in graph.nodes
    ]
    temporal_idx = names.index("temporal") if "temporal" in names else -1
    for stage in ("inference", "reflection", "save"):
        if stage in names:
            assert names.index(stage) > temporal_idx, (
                f"Stage '{stage}' must come after 'temporal' in pipeline; "
                f"got temporal={temporal_idx}, {stage}={names.index(stage)}"
            )


def test_turn_context_mutation_queue_bound_on_construction():
    """TurnContext.__post_init__ must bind the MutationQueue to the turn's trace_id."""
    graph_mod = importlib.import_module("dadbot.core.graph")
    ctx = graph_mod.TurnContext(user_input="integrity test")
    assert ctx.mutation_queue._owner_trace_id == ctx.trace_id, (
        "MutationQueue must be bound to TurnContext.trace_id at construction"
    )


def test_turn_graph_refuses_duplicate_stage_execution():
    """TurnGraph must raise RuntimeError when a stage is executed twice."""
    import asyncio
    graph_mod = importlib.import_module("dadbot.core.graph")

    executed: list[str] = []

    class _DupNode:
        name = "temporal"

        async def execute(self, _registry: Any, turn_context: Any) -> None:
            executed.append("temporal")

    graph = graph_mod.TurnGraph()
    ctx = graph_mod.TurnContext(user_input="dup test")

    with pytest.raises(RuntimeError, match="executed more than once"):
        graph._mark_stage_enter(ctx, "temporal")
        graph._mark_stage_enter(ctx, "temporal")


# ---------------------------------------------------------------------------
# D. Persistence contract invariants
# ---------------------------------------------------------------------------


def test_persistence_service_contract_fields():
    """PersistenceServiceContract must expose version, save_turn, save_graph_checkpoint, save_turn_event."""
    policy_mod = importlib.import_module("dadbot.core.execution_policy")
    contract = policy_mod.PersistenceServiceContract()
    assert contract.version, "PersistenceServiceContract.version must be non-empty"
    assert contract.save_turn == "save_turn"
    assert contract.save_graph_checkpoint == "save_graph_checkpoint"
    assert contract.save_turn_event == "save_turn_event"


def test_persistence_contract_strict_mode_raises_on_missing_service():
    """validate_persistence_service_contract must raise in strict mode when service is None."""
    policy_mod = importlib.import_module("dadbot.core.execution_policy")
    engine = policy_mod.ExecutionPolicyEngine(
        persistence_contract=policy_mod.PersistenceServiceContract()
    )
    with pytest.raises(RuntimeError, match="unavailable"):
        engine.validate_persistence_service_contract(None, strict_mode=True)


def test_persistence_contract_lenient_mode_returns_ok_false_for_missing_service():
    """validate_persistence_service_contract must return ok=False (not raise) in lenient mode."""
    policy_mod = importlib.import_module("dadbot.core.execution_policy")
    engine = policy_mod.ExecutionPolicyEngine(
        persistence_contract=policy_mod.PersistenceServiceContract()
    )
    result = engine.validate_persistence_service_contract(None, strict_mode=False)
    assert result["ok"] is False
    assert len(result["missing"]) > 0


def test_persistence_contract_strict_mode_raises_on_missing_methods():
    """validate_persistence_service_contract must raise in strict mode when required methods absent."""
    policy_mod = importlib.import_module("dadbot.core.execution_policy")
    engine = policy_mod.ExecutionPolicyEngine(
        persistence_contract=policy_mod.PersistenceServiceContract()
    )
    incomplete_service = SimpleNamespace(save_turn=lambda *a, **k: None)  # missing other methods
    with pytest.raises(RuntimeError, match="contract violation"):
        engine.validate_persistence_service_contract(incomplete_service, strict_mode=True)


def test_persistence_contract_passes_for_full_service():
    """validate_persistence_service_contract must return ok=True for a complete service."""
    policy_mod = importlib.import_module("dadbot.core.execution_policy")
    engine = policy_mod.ExecutionPolicyEngine(
        persistence_contract=policy_mod.PersistenceServiceContract()
    )
    full_service = SimpleNamespace(
        save_turn=lambda *a, **k: None,
        save_graph_checkpoint=lambda *a, **k: None,
        save_turn_event=lambda *a, **k: None,
    )
    result = engine.validate_persistence_service_contract(full_service, strict_mode=True)
    assert result["ok"] is True
    assert result["missing"] == []


def test_persistence_contract_strict_mode_raises_on_invalid_method_signature():
    """Strict mode must reject persistence services with invalid required method arity."""
    policy_mod = importlib.import_module("dadbot.core.execution_policy")
    engine = policy_mod.ExecutionPolicyEngine(
        persistence_contract=policy_mod.PersistenceServiceContract()
    )
    malformed_service = SimpleNamespace(
        save_turn=lambda _context: None,
        save_graph_checkpoint=lambda _checkpoint: None,
        save_turn_event=lambda _event: None,
    )
    with pytest.raises(RuntimeError, match="signature_issues"):
        engine.validate_persistence_service_contract(malformed_service, strict_mode=True)


def test_persistence_contract_lenient_mode_surfaces_signature_issues():
    """Lenient mode should return signature diagnostics without raising."""
    policy_mod = importlib.import_module("dadbot.core.execution_policy")
    engine = policy_mod.ExecutionPolicyEngine(
        persistence_contract=policy_mod.PersistenceServiceContract()
    )
    malformed_service = SimpleNamespace(
        save_turn=lambda _context: None,
        save_graph_checkpoint=lambda _checkpoint: None,
        save_turn_event=lambda _event: None,
    )
    result = engine.validate_persistence_service_contract(malformed_service, strict_mode=False)
    assert result["ok"] is False
    assert any("save_turn" in issue for issue in list(result.get("signature_issues") or []))


# ---------------------------------------------------------------------------
# E. ExecutionIdentity contract (GAP-1/GAP-2 enforcement)
# ---------------------------------------------------------------------------


def test_execution_identity_fingerprint_is_deterministic():
    """Same inputs must produce the same ExecutionIdentity fingerprint."""
    identity_mod = importlib.import_module("dadbot.core.execution_identity")
    kwargs = dict(
        trace_id="abc123",
        trace_hash="deadbeef",
        lock_hash="cafebabe",
        checkpoint_chain_hash="11223344",
        mutation_tx_count=2,
        event_count=7,
    )
    a = identity_mod.ExecutionIdentity(**kwargs)
    b = identity_mod.ExecutionIdentity(**kwargs)
    assert a.fingerprint == b.fingerprint, "ExecutionIdentity fingerprint must be deterministic"


def test_execution_identity_raise_if_mismatch_raises_on_wrong_fingerprint():
    """raise_if_mismatch must raise ExecutionIdentityViolation when fingerprint differs."""
    identity_mod = importlib.import_module("dadbot.core.execution_identity")
    identity = identity_mod.ExecutionIdentity(
        trace_id="t1",
        trace_hash="h1",
        lock_hash="l1",
        checkpoint_chain_hash="c1",
        mutation_tx_count=0,
        event_count=1,
    )
    with pytest.raises(identity_mod.ExecutionIdentityViolation) as exc_info:
        identity.raise_if_mismatch("not_the_right_fingerprint")
    assert exc_info.value.expected == "not_the_right_fingerprint"
    assert exc_info.value.actual == identity.fingerprint


def test_execution_identity_raise_if_mismatch_no_op_when_empty():
    """raise_if_mismatch must be a no-op when expected is empty (no contract set)."""
    identity_mod = importlib.import_module("dadbot.core.execution_identity")
    identity = identity_mod.ExecutionIdentity(
        trace_id="t1",
        trace_hash="h1",
        lock_hash="l1",
        checkpoint_chain_hash="c1",
        mutation_tx_count=0,
        event_count=1,
    )
    identity.raise_if_mismatch("")  # must not raise


def test_execution_identity_to_dict_includes_fingerprint():
    """to_dict() must include the 'fingerprint' key."""
    identity_mod = importlib.import_module("dadbot.core.execution_identity")
    identity = identity_mod.ExecutionIdentity(
        trace_id="t1",
        trace_hash="h1",
        lock_hash="l1",
        checkpoint_chain_hash="c1",
        mutation_tx_count=1,
        event_count=3,
    )
    d = identity.to_dict()
    assert "fingerprint" in d, "ExecutionIdentity.to_dict() must include 'fingerprint'"
    assert d["fingerprint"] == identity.fingerprint


# ---------------------------------------------------------------------------
# F. Stage-phase mapping policy (GAP-3 enforcement)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "stage,expected_phase",
    [
        ("temporal", "PLAN"),
        ("health", "PLAN"),
        ("memory", "PLAN"),
        ("context", "PLAN"),
        ("plan", "PLAN"),
        ("inference", "ACT"),
        ("agent", "ACT"),
        ("tool", "ACT"),
        ("act", "ACT"),
        ("safety", "OBSERVE"),
        ("guard", "OBSERVE"),
        ("observe", "OBSERVE"),
        ("moderation", "OBSERVE"),
        ("save", "RESPOND"),
        ("respond", "RESPOND"),
        ("persist", "RESPOND"),
        ("finalize", "RESPOND"),
    ],
)
def test_stage_phase_mapping_policy(stage: str, expected_phase: str):
    """StagePhaseMappingPolicy must return the canonical phase for every known stage."""
    policy_mod = importlib.import_module("dadbot.core.execution_policy")
    result = policy_mod.StagePhaseMappingPolicy.phase_name_for_stage(stage)
    assert result == expected_phase, (
        f"StagePhaseMappingPolicy.phase_name_for_stage({stage!r}) returned {result!r}, "
        f"expected {expected_phase!r}"
    )


def test_stage_phase_mapping_unknown_stage_returns_empty():
    """Unknown stages must return an empty string (not raise)."""
    policy_mod = importlib.import_module("dadbot.core.execution_policy")
    result = policy_mod.StagePhaseMappingPolicy.phase_name_for_stage("completely_unknown_stage")
    assert result == "", (
        f"Unknown stages must map to empty string, got {result!r}"
    )


def test_stage_phase_mapping_audit_map_covers_all_stages():
    """all_stage_phase_map() must return a non-empty dict covering all known stages."""
    policy_mod = importlib.import_module("dadbot.core.execution_policy")
    full_map = policy_mod.StagePhaseMappingPolicy.all_stage_phase_map()
    assert isinstance(full_map, dict) and len(full_map) > 0
    assert all(v in ("PLAN", "ACT", "OBSERVE", "RESPOND") for v in full_map.values()), (
        "all_stage_phase_map() must only produce PLAN/ACT/OBSERVE/RESPOND values"
    )


# ---------------------------------------------------------------------------
# STRICT-mode bonus: assert no unguarded global mutation_queue.queue() calls
# outside of SaveNode-equivalent context in non-test, non-graph source files.
# This is a WARN in normal mode and HARD FAIL in PHASE4_STRICT=1 mode.
# ---------------------------------------------------------------------------

_MUTATION_QUEUE_OWNERS = frozenset(
    {
        "dadbot/core/graph.py",          # owns MutationGuard
        "dadbot/services/turn_service.py",  # SaveNode executor — legitimate caller
    }
)


def test_mutation_queue_callers_are_approved():
    """Files that call mutation_queue.queue() must be in the approved allow-list.

    In PHASE4_STRICT=1 mode this is a hard failure; otherwise it's a warning.
    """
    unapproved: list[str] = []
    for py_file in DADBOT_SRC.rglob("*.py"):
        rel = _rel(py_file)
        if any(rel.endswith(a.lstrip("./")) for a in _MUTATION_QUEUE_OWNERS):
            continue
        text = py_file.read_text(encoding="utf-8", errors="replace")
        if "mutation_queue.queue(" in text:
            unapproved.append(rel)

    if unapproved:
        msg = (
            "Files calling mutation_queue.queue() outside approved callers:\n"
            + "\n".join(f"  {f}" for f in unapproved)
        )
        if STRICT:
            pytest.fail(msg)
        else:
            pytest.warns(UserWarning) if False else None  # pragma: no cover
            import warnings
            warnings.warn(msg, stacklevel=2)
