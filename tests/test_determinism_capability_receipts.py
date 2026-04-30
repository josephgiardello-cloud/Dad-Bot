"""Tests for Tier 1 (Determinism Edge Sealing), Tier 2 (Capability Security),
Tier 3 (Execution Receipts), and the side-effect deduplication fix.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Side-Effect Deduplication Tests (Critical Fix)
# ---------------------------------------------------------------------------


class TestSideEffectDeduplication:
    """in_flight_stage marker prevents duplicate side effects on crash-resume."""

    def test_mark_started_persists_in_flight_marker(self, tmp_path: Path) -> None:
        from dadbot.core.turn_resume_store import TurnResumeStore

        store = TurnResumeStore(tmp_path)
        # No record yet — mark_started creates a minimal one.
        store.mark_started("t1", "inference")
        record = store.load("t1")
        assert record is not None
        assert record.in_flight_stage == "inference"

    def test_mark_started_updates_existing_record(self, tmp_path: Path) -> None:
        from dadbot.core.turn_resume_store import TurnResumeStore

        store = TurnResumeStore(tmp_path)
        store.save(
            turn_id="t2",
            last_completed_stage="temporal",
            next_stage="health",
            checkpoint_hash="aa" * 16,
            completed_stages=["temporal"],
        )
        store.mark_started("t2", "health")
        record = store.load("t2")
        assert record is not None
        assert record.in_flight_stage == "health"
        # Completed stages are preserved.
        assert "temporal" in record.completed_stages

    def test_completed_save_clears_in_flight(self, tmp_path: Path) -> None:
        from dadbot.core.turn_resume_store import TurnResumeStore

        store = TurnResumeStore(tmp_path)
        store.mark_started("t3", "inference")
        store.save(
            turn_id="t3",
            last_completed_stage="inference",
            next_stage="safety",
            checkpoint_hash="bb" * 16,
            completed_stages=["temporal", "inference"],
            # in_flight_stage is not part of save() signature — it defaults to ""
        )
        record = store.load("t3")
        assert record is not None
        # save() always writes in_flight_stage="" because ResumePoint default is ""
        assert record.in_flight_stage == ""

    def test_resume_point_round_trips_in_flight(self, tmp_path: Path) -> None:
        from dadbot.core.turn_resume_store import ResumePoint, TurnResumeStore

        store = TurnResumeStore(tmp_path)
        store.mark_started("t4", "safety")
        record = store.load("t4")
        assert record is not None
        # Round-trip via dict
        reconstructed = ResumePoint.from_dict(record.to_dict())
        assert reconstructed.in_flight_stage == "safety"

    def test_execution_recovery_injects_stage_call_id(self, tmp_path: Path) -> None:
        from dadbot.core.execution_policy import ResumabilityPolicy
        from dadbot.core.execution_recovery import ExecutionRecovery, _compute_stage_call_id
        from dadbot.core.turn_resume_store import TurnResumeStore

        store = TurnResumeStore(tmp_path)
        policy = ResumabilityPolicy()
        recovery = ExecutionRecovery(store, policy)

        ctx = MagicMock()
        ctx.trace_id = "myturn"
        ctx.state = {}
        call_id = recovery.mark_stage_started("inference", ctx)

        # call_id returned must match deterministic formula.
        expected = _compute_stage_call_id("myturn", "inference")
        assert call_id == expected

        # State must have _stage_call_id injected.
        assert ctx.state.get("_stage_call_id") == expected

    def test_stage_call_id_is_deterministic(self) -> None:
        from dadbot.core.execution_recovery import _compute_stage_call_id

        id1 = _compute_stage_call_id("turn-abc", "inference")
        id2 = _compute_stage_call_id("turn-abc", "inference")
        assert id1 == id2

    def test_stage_call_id_differs_by_stage(self) -> None:
        from dadbot.core.execution_recovery import _compute_stage_call_id

        assert _compute_stage_call_id("t", "inference") != _compute_stage_call_id("t", "safety")

    def test_stage_call_id_differs_by_turn(self) -> None:
        from dadbot.core.execution_recovery import _compute_stage_call_id

        assert _compute_stage_call_id("turn-A", "inference") != _compute_stage_call_id("turn-B", "inference")


# ---------------------------------------------------------------------------
# Tier 1 — Float Normalizer
# ---------------------------------------------------------------------------


class TestFloatNormalizer:
    def test_rounds_float_to_precision(self) -> None:
        from dadbot.core.determinism_seal import FloatNormalizer

        n = FloatNormalizer(precision=4)
        assert n.normalize(1.123456789) == 1.1235

    def test_normalizes_nested_dict(self) -> None:
        from dadbot.core.determinism_seal import FloatNormalizer

        n = FloatNormalizer(precision=2)
        result = n.normalize({"a": 1.999, "b": {"c": 0.004999}})
        assert result == {"a": 2.0, "b": {"c": 0.0}}

    def test_normalizes_list(self) -> None:
        from dadbot.core.determinism_seal import FloatNormalizer

        n = FloatNormalizer(precision=3)
        result = n.normalize([1.1119, 2.2225])
        assert result[0] == round(1.1119, 3)
        assert result[1] == round(2.2225, 3)

    def test_non_float_unchanged(self) -> None:
        from dadbot.core.determinism_seal import FloatNormalizer

        n = FloatNormalizer()
        assert n.normalize("hello") == "hello"
        assert n.normalize(42) == 42

    def test_nan_and_inf_become_zero(self) -> None:
        from dadbot.core.determinism_seal import FloatNormalizer

        n = FloatNormalizer()
        assert n.normalize(float("nan")) == 0.0
        assert n.normalize(float("inf")) == 0.0
        assert n.normalize(float("-inf")) == 0.0

    def test_tuple_preserved_as_tuple(self) -> None:
        from dadbot.core.determinism_seal import FloatNormalizer

        n = FloatNormalizer(precision=1)
        result = n.normalize((1.15, 2.25))
        assert isinstance(result, tuple)
        # Values must be rounded to 1 decimal place (Python uses banker's rounding).
        assert result == (round(1.15, 1), round(2.25, 1))


# ---------------------------------------------------------------------------
# Tier 1 — Tool Output Normalizer
# ---------------------------------------------------------------------------


class TestToolOutputNormalizer:
    def test_normalizes_crlf_to_lf(self) -> None:
        from dadbot.core.determinism_seal import ToolOutputNormalizer

        n = ToolOutputNormalizer()
        assert n.normalize("hello\r\nworld") == "hello\nworld"

    def test_strips_trailing_blank_lines(self) -> None:
        from dadbot.core.determinism_seal import ToolOutputNormalizer

        n = ToolOutputNormalizer()
        assert n.normalize("hello\n\n\n") == "hello"

    def test_collapses_whitespace_runs(self) -> None:
        from dadbot.core.determinism_seal import ToolOutputNormalizer

        n = ToolOutputNormalizer()
        assert n.normalize("hello   world") == "hello world"

    def test_strips_line_leading_trailing_whitespace(self) -> None:
        from dadbot.core.determinism_seal import ToolOutputNormalizer

        n = ToolOutputNormalizer()
        assert n.normalize("  hello  \n  world  ") == "hello\nworld"

    def test_non_string_unchanged(self) -> None:
        from dadbot.core.determinism_seal import ToolOutputNormalizer

        n = ToolOutputNormalizer()
        assert n.normalize(42) == 42
        assert n.normalize(None) is None

    def test_normalize_dict_recurses(self) -> None:
        from dadbot.core.determinism_seal import ToolOutputNormalizer

        n = ToolOutputNormalizer()
        result = n.normalize_dict({"key": "  value  \r\n"})
        assert result == {"key": "value"}


# ---------------------------------------------------------------------------
# Tier 1 — Time Leak Detector
# ---------------------------------------------------------------------------


class TestTimeLeakDetector:
    def test_no_leaks_when_state_unchanged(self) -> None:
        from dadbot.core.determinism_seal import TimeLeakDetector

        d = TimeLeakDetector()
        state = {"score": 0.9, "ts": 1_700_000_000.0}
        d.snapshot(state)
        leaks = d.find_leaks(state)  # same state
        assert leaks == []

    def test_detects_new_time_like_float(self) -> None:
        from dadbot.core.determinism_seal import TimeLeakDetector

        d = TimeLeakDetector()
        before = {"score": 0.9}
        d.snapshot(before)
        after = {"score": 0.9, "leaked_ts": 1_700_000_001.0}
        leaks = d.find_leaks(after)
        assert len(leaks) == 1
        assert "leaked_ts" in leaks[0]

    def test_no_false_positive_for_small_values(self) -> None:
        from dadbot.core.determinism_seal import TimeLeakDetector

        d = TimeLeakDetector()
        before = {"small": 0.5}
        d.snapshot(before)
        after = {"small": 0.6}  # not time-like
        leaks = d.find_leaks(after)
        assert leaks == []


# ---------------------------------------------------------------------------
# Tier 1 — Randomness Seal
# ---------------------------------------------------------------------------


class TestRandomnessSeal:
    def test_detects_import_random(self) -> None:
        from dadbot.core.determinism_seal import RandomnessSeal

        s = RandomnessSeal()
        violations = s.audit_source("import random\nx = random.random()", filename="node.py")
        assert any("random" in v for v in violations)

    def test_detects_from_random_import(self) -> None:
        from dadbot.core.determinism_seal import RandomnessSeal

        s = RandomnessSeal()
        violations = s.audit_source("from random import choice", filename="node.py")
        assert len(violations) >= 1

    def test_clean_code_has_no_violations(self) -> None:
        from dadbot.core.determinism_seal import RandomnessSeal

        s = RandomnessSeal()
        clean = "import hashlib\nx = hashlib.sha256(b'data').hexdigest()"
        assert s.audit_source(clean) == []


# ---------------------------------------------------------------------------
# Tier 1 — DeterminismSeal (unified)
# ---------------------------------------------------------------------------


class TestDeterminismSeal:
    def test_apply_normalizes_floats(self) -> None:
        from dadbot.core.determinism_seal import DeterminismSeal, DeterminismSealConfig

        seal = DeterminismSeal(DeterminismSealConfig(float_precision=3))
        result = seal.apply({"latency_ms": 123.456789})
        assert result["latency_ms"] == 123.457

    def test_apply_normalizes_strings(self) -> None:
        from dadbot.core.determinism_seal import DeterminismSeal

        seal = DeterminismSeal()
        result = seal.apply({"response": "  hello  \r\n"})
        assert result["response"] == "hello"

    def test_default_seal_is_importable(self) -> None:
        from dadbot.core.determinism_seal import DEFAULT_SEAL

        assert DEFAULT_SEAL is not None

    def test_snapshot_and_find_time_leaks(self) -> None:
        from dadbot.core.determinism_seal import DeterminismSeal

        seal = DeterminismSeal()
        seal.snapshot_for_leak_detection({"x": 0.5})
        leaks = seal.find_time_leaks({"x": 0.5, "ts": 1_700_000_009.0})
        assert len(leaks) == 1

    def test_audit_for_randomness_delegates(self) -> None:
        from dadbot.core.determinism_seal import DeterminismSeal

        violations = DeterminismSeal.audit_for_randomness("import random")
        assert len(violations) >= 1


# ---------------------------------------------------------------------------
# Tier 2 — CapabilityRegistry
# ---------------------------------------------------------------------------


class TestCapabilityRegistry:
    def test_register_and_lookup(self) -> None:
        from dadbot.core.authorization import Capability
        from dadbot.core.capability_registry import (
            CapabilityRegistry,
            EnforcementMode,
            NodeCapabilityRequirement,
        )

        reg = CapabilityRegistry()
        req = NodeCapabilityRequirement(
            stage="inference",
            required_capabilities=frozenset({Capability.EXECUTE}),
            mode=EnforcementMode.ENFORCE,
        )
        reg.register("inference", req)
        found = reg.requirement_for("inference")
        assert found.stage == "inference"
        assert Capability.EXECUTE in found.required_capabilities

    def test_fallback_to_global_default(self) -> None:
        from dadbot.core.capability_registry import CapabilityRegistry

        reg = CapabilityRegistry()
        req = reg.requirement_for("unknown_stage")
        assert req.stage == "*"
        assert not req.required_capabilities

    def test_is_satisfied_by_full_capset(self) -> None:
        from dadbot.core.authorization import Capability, CapabilitySet
        from dadbot.core.capability_registry import NodeCapabilityRequirement

        req = NodeCapabilityRequirement(
            stage="save",
            required_capabilities=frozenset({Capability.WRITE}),
        )
        assert req.is_satisfied_by(CapabilitySet.full())
        assert not req.is_satisfied_by(CapabilitySet.read_only())


class TestEnforceNodeEntry:
    def test_passes_when_satisfied(self) -> None:
        from dadbot.core.authorization import Capability, CapabilitySet
        from dadbot.core.capability_registry import (
            CapabilityRegistry,
            EnforcementMode,
            NodeCapabilityRequirement,
            enforce_node_entry,
        )

        reg = CapabilityRegistry()
        reg.register(
            "inference",
            NodeCapabilityRequirement(
                stage="inference",
                required_capabilities=frozenset({Capability.EXECUTE}),
                mode=EnforcementMode.ENFORCE,
            ),
        )
        mode = enforce_node_entry("inference", registry=reg, caps=CapabilitySet.full())
        assert mode == EnforcementMode.ENFORCE

    def test_raises_on_missing_capability(self) -> None:
        from dadbot.core.authorization import Capability, CapabilitySet
        from dadbot.core.capability_registry import (
            CapabilityRegistry,
            CapabilityViolationError,
            EnforcementMode,
            NodeCapabilityRequirement,
            enforce_node_entry,
        )

        reg = CapabilityRegistry()
        reg.register(
            "inference",
            NodeCapabilityRequirement(
                stage="inference",
                required_capabilities=frozenset({Capability.EXECUTE}),
                mode=EnforcementMode.ENFORCE,
            ),
        )
        with pytest.raises(CapabilityViolationError):
            enforce_node_entry("inference", registry=reg, caps=CapabilitySet.read_only())

    def test_warn_mode_does_not_raise(self) -> None:
        from dadbot.core.authorization import Capability, CapabilitySet
        from dadbot.core.capability_registry import (
            CapabilityRegistry,
            EnforcementMode,
            NodeCapabilityRequirement,
            enforce_node_entry,
        )

        reg = CapabilityRegistry()
        reg.register(
            "inference",
            NodeCapabilityRequirement(
                stage="inference",
                required_capabilities=frozenset({Capability.EXECUTE}),
                mode=EnforcementMode.WARN,
            ),
        )
        mode = enforce_node_entry("inference", registry=reg, caps=CapabilitySet.read_only())
        assert mode == EnforcementMode.WARN

    def test_skip_mode_returns_skip(self) -> None:
        from dadbot.core.authorization import Capability, CapabilitySet
        from dadbot.core.capability_registry import (
            CapabilityRegistry,
            EnforcementMode,
            NodeCapabilityRequirement,
            enforce_node_entry,
        )

        reg = CapabilityRegistry()
        reg.register(
            "save",
            NodeCapabilityRequirement(
                stage="save",
                required_capabilities=frozenset({Capability.WRITE}),
                mode=EnforcementMode.SKIP,
            ),
        )
        mode = enforce_node_entry("save", registry=reg, caps=CapabilitySet.read_only())
        assert mode == EnforcementMode.SKIP

    def test_no_caps_is_noop(self) -> None:
        from dadbot.core.capability_registry import CapabilityRegistry, enforce_node_entry

        reg = CapabilityRegistry()
        mode = enforce_node_entry("inference", registry=reg, caps=None)
        assert mode is not None  # does not raise


class TestCapabilitySnapshot:
    def test_from_policy_captures_granted_caps(self) -> None:
        from dadbot.core.authorization import Capability, CapabilitySet, SessionAuthorizationPolicy
        from dadbot.core.capability_registry import CapabilitySnapshot

        policy = SessionAuthorizationPolicy()
        policy.grant("sess-1", CapabilitySet(Capability.READ, Capability.WRITE))
        snap = CapabilitySnapshot.from_policy("sess-1", policy)
        assert Capability.READ.value in snap.granted
        assert Capability.WRITE.value in snap.granted
        assert Capability.EXECUTE.value not in snap.granted

    def test_is_escalation_detects_new_caps(self) -> None:
        from dadbot.core.capability_registry import CapabilitySnapshot

        original = CapabilitySnapshot(session_id="s", granted=("read",))
        escalated = CapabilitySnapshot(session_id="s2", granted=("read", "write"))
        assert escalated.is_escalation_of(original)

    def test_is_escalation_false_when_equal(self) -> None:
        from dadbot.core.capability_registry import CapabilitySnapshot

        snap = CapabilitySnapshot(session_id="s", granted=("read", "write"))
        assert not snap.is_escalation_of(snap)

    def test_freeze_and_verify_no_escalation(self) -> None:
        from dadbot.core.authorization import Capability, CapabilitySet, SessionAuthorizationPolicy
        from dadbot.core.capability_registry import freeze_capabilities, verify_capability_freeze

        policy = SessionAuthorizationPolicy()
        policy.grant("s", CapabilitySet(Capability.READ))
        ctx = MagicMock()
        ctx.state = {}
        freeze_capabilities(ctx, policy=policy, session_id="s")
        # Same session, same caps — should not raise.
        verify_capability_freeze(ctx, policy=policy, session_id="s")

    def test_freeze_and_verify_raises_on_escalation(self) -> None:
        from dadbot.core.authorization import Capability, CapabilitySet, SessionAuthorizationPolicy
        from dadbot.core.capability_registry import (
            CapabilityViolationError,
            freeze_capabilities,
            verify_capability_freeze,
        )

        policy = SessionAuthorizationPolicy()
        policy.grant("s-original", CapabilitySet(Capability.READ))
        ctx = MagicMock()
        ctx.state = {}
        # Freeze with limited caps.
        freeze_capabilities(ctx, policy=policy, session_id="s-original")
        # Now grant new session admin caps.
        policy.grant("s-admin", CapabilitySet.full())
        with pytest.raises(CapabilityViolationError, match="escalation"):
            verify_capability_freeze(ctx, policy=policy, session_id="s-admin")

    def test_round_trip_via_dict(self) -> None:
        from dadbot.core.capability_registry import CapabilitySnapshot

        snap = CapabilitySnapshot(session_id="s", granted=("execute", "read", "write"))
        reconstructed = CapabilitySnapshot.from_dict(snap.to_dict())
        assert reconstructed == snap

    def test_configure_capabilities_sets_registry_on_graph(self) -> None:
        from dadbot.core.authorization import CapabilitySet, SessionAuthorizationPolicy
        from dadbot.core.capability_registry import CapabilityRegistry
        from dadbot.core.graph import TurnGraph

        graph = TurnGraph()
        reg = CapabilityRegistry()
        policy = SessionAuthorizationPolicy()
        policy.grant("s", CapabilitySet.full())
        graph.configure_capabilities(reg, policy=policy, session_id="s")
        assert graph._capability_registry is reg
        assert graph._capability_policy is policy


# ---------------------------------------------------------------------------
# Tier 3 — ExecutionReceipt
# ---------------------------------------------------------------------------


class TestExecutionReceipt:
    def test_sign_and_verify(self) -> None:
        from dadbot.core.execution_receipt import ReceiptSigner

        signer = ReceiptSigner(secret_key=b"k" * 32)
        receipt = signer.sign(
            turn_id="t",
            stage="inference",
            sequence=1,
            stage_call_id="abc",
            checkpoint_hash="deadbeef",
            prev_receipt_sig="",
        )
        assert receipt.signature
        assert signer.verify(receipt)

    def test_tampered_receipt_fails_verification(self) -> None:
        from dadbot.core.execution_receipt import ExecutionReceipt, ReceiptSigner

        signer = ReceiptSigner(secret_key=b"k" * 32)
        receipt = signer.sign(
            turn_id="t",
            stage="inference",
            sequence=1,
            stage_call_id="abc",
            checkpoint_hash="deadbeef",
            prev_receipt_sig="",
        )
        # Tamper with the checkpoint_hash.
        tampered = ExecutionReceipt(
            turn_id=receipt.turn_id,
            stage=receipt.stage,
            sequence=receipt.sequence,
            stage_call_id=receipt.stage_call_id,
            checkpoint_hash="tampered",
            prev_receipt_sig=receipt.prev_receipt_sig,
            completed_at=receipt.completed_at,
            signature=receipt.signature,
        )
        assert not signer.verify(tampered)

    def test_wrong_key_fails_verification(self) -> None:
        from dadbot.core.execution_receipt import ReceiptSigner

        signer1 = ReceiptSigner(secret_key=b"key1" + b"\x00" * 28)
        signer2 = ReceiptSigner(secret_key=b"key2" + b"\x00" * 28)
        receipt = signer1.sign(
            turn_id="t",
            stage="save",
            sequence=3,
            stage_call_id="xyz",
            checkpoint_hash="hash",
            prev_receipt_sig="prev",
        )
        assert not signer2.verify(receipt)

    def test_round_trip_via_dict(self) -> None:
        from dadbot.core.execution_receipt import ExecutionReceipt, ReceiptSigner

        signer = ReceiptSigner(secret_key=b"k" * 32)
        receipt = signer.sign(
            turn_id="t",
            stage="save",
            sequence=5,
            stage_call_id="xyz",
            checkpoint_hash="aa" * 16,
            prev_receipt_sig="prevhex",
        )
        reconstructed = ExecutionReceipt.from_dict(receipt.to_dict())
        assert reconstructed == receipt


# ---------------------------------------------------------------------------
# Tier 3 — ReceiptChain
# ---------------------------------------------------------------------------


class TestReceiptChain:
    def _build_chain(self, signer, stages: list[str], turn_id: str = "t") -> ReceiptChain:
        from dadbot.core.execution_receipt import ReceiptChain

        chain = ReceiptChain()
        for stage in stages:
            receipt = signer.sign(
                turn_id=turn_id,
                stage=stage,
                sequence=chain.next_sequence,
                stage_call_id=f"call_{stage}",
                checkpoint_hash=f"hash_{stage}",
                prev_receipt_sig=chain.last_signature,
            )
            chain.append(receipt)
        return chain

    def test_intact_chain_has_no_violations(self) -> None:
        from dadbot.core.execution_receipt import ReceiptSigner

        signer = ReceiptSigner(secret_key=b"k" * 32)
        chain = self._build_chain(signer, ["temporal", "health", "inference", "save"])
        violations = chain.verify_continuity(signer, expected_stages=["temporal", "inference", "save"])
        assert violations == []

    def test_missing_expected_stage_detected(self) -> None:
        from dadbot.core.execution_receipt import ReceiptSigner

        signer = ReceiptSigner(secret_key=b"k" * 32)
        chain = self._build_chain(signer, ["temporal", "health"])
        violations = chain.verify_continuity(signer, expected_stages=["temporal", "inference"])
        assert any("inference" in v.reason for v in violations)

    def test_tampered_receipt_detected(self) -> None:
        from dadbot.core.execution_receipt import ExecutionReceipt, ReceiptChain, ReceiptSigner

        signer = ReceiptSigner(secret_key=b"k" * 32)
        chain = self._build_chain(signer, ["temporal", "inference"])
        # Replace inference receipt with tampered version.
        original = chain.receipts[1]
        tampered = ExecutionReceipt(
            turn_id=original.turn_id,
            stage=original.stage,
            sequence=original.sequence,
            stage_call_id=original.stage_call_id,
            checkpoint_hash="TAMPERED",
            prev_receipt_sig=original.prev_receipt_sig,
            completed_at=original.completed_at,
            signature=original.signature,  # signature is now invalid
        )
        chain2 = ReceiptChain([chain.receipts[0], tampered])
        violations = chain2.verify_continuity(signer)
        assert len(violations) >= 1

    def test_state_round_trip(self) -> None:
        from dadbot.core.execution_receipt import ReceiptChain, ReceiptSigner

        signer = ReceiptSigner(secret_key=b"k" * 32)
        chain = self._build_chain(signer, ["temporal", "health", "save"])
        state: dict = {}
        chain.write_to_state(state)
        loaded = ReceiptChain.from_state(state)
        assert len(loaded) == 3

    def test_empty_chain_is_intact(self) -> None:
        from dadbot.core.execution_receipt import ReceiptChain, ReceiptSigner

        signer = ReceiptSigner(secret_key=b"k" * 32)
        chain = ReceiptChain()
        assert chain.is_intact(signer)

    def test_next_sequence_increments(self) -> None:
        from dadbot.core.execution_receipt import ReceiptChain, ReceiptSigner

        signer = ReceiptSigner(secret_key=b"k" * 32)
        chain = ReceiptChain()
        assert chain.next_sequence == 1
        chain = self._build_chain(signer, ["temporal"])
        assert chain.next_sequence == 2

    def test_configure_receipt_signer_on_graph(self) -> None:
        from dadbot.core.execution_receipt import ReceiptSigner
        from dadbot.core.graph import TurnGraph

        graph = TurnGraph()
        signer = ReceiptSigner(secret_key=b"stable" + b"\x00" * 26)
        graph.configure_receipt_signer(signer)
        assert graph._receipt_signer is signer
