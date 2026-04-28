"""Part 2 — Execution Truth Binding Test.

THE SINGLE TEST THAT PROVES REALITY VS CLAIMS.

This test answers the question:
  "What did the system ACTUALLY do vs what it CLAIMS it did?"

It compares:
  CLAIM LAYER   — what the orchestrator/planner reports happened
  EVIDENCE LAYER — what execution receipts + tool IR + memory logs show

If these layers diverge, the system is internally inconsistent:
  * Reporting metrics that don't correspond to real execution
  * Silently skipping stages while claiming they ran
  * Hallucinating tool calls that never happened

Design
------
The test runs at three levels:
  Level 1 — Pure unit: synthetic state → claim vs evidence alignment
  Level 2 — Mutation: deliberate divergence detects violations correctly
  Level 3 — Structural: receipt chain integrity binds both layers

All levels are fully offline (no LLM, no network, no filesystem I/O).
"""
from __future__ import annotations

import hashlib
from typing import Any

import pytest

from dadbot.uril.truth_binding import (
    BindingViolation,
    ClaimBindingResult,
    ClaimEvidenceValidator,
    ExecutionClaim,
    ExecutionEvidence,
    build_synthetic_state,
    compute_receipt_chain_hash,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_full_state(
    turn_id: str = "truth-test-001",
    stages: list[str] | None = None,
    tools: list[str] | None = None,
    memory_keys: list[str] | None = None,
) -> dict[str, Any]:
    """Build a realistic, self-consistent TurnContext.state dict."""
    return build_synthetic_state(
        turn_id=turn_id,
        stages=stages or ["plan", "temporal", "inference", "tool", "memory", "respond"],
        tools=tools or ["search_web", "query_memory"],
        memory_keys=memory_keys or ["consolidated_memories", "relationship_history", "life_patterns"],
    )


# ---------------------------------------------------------------------------
# Level 1: Unit — claim == evidence on consistent state
# ---------------------------------------------------------------------------

class TestTruthBindingUnit:
    """Claim and evidence must agree when state is internally consistent."""

    def test_binding_valid_on_clean_state(self):
        state = _make_full_state()
        validator = ClaimEvidenceValidator()
        claim = validator.extract_claim_from_state(state, "truth-test-001")
        evidence = validator.extract_evidence_from_state(state, "truth-test-001")
        result = validator.validate(claim, evidence)
        assert result.valid, f"Expected clean binding. Violations: {result.to_dict()}"

    def test_claim_turn_id_matches_evidence(self):
        state = _make_full_state(turn_id="turn-abc-123")
        validator = ClaimEvidenceValidator()
        claim = validator.extract_claim_from_state(state, "turn-abc-123")
        evidence = validator.extract_evidence_from_state(state, "turn-abc-123")
        assert claim.turn_id == evidence.turn_id == "turn-abc-123"

    def test_claim_receipt_hash_matches_evidence(self):
        state = _make_full_state()
        validator = ClaimEvidenceValidator()
        claim = validator.extract_claim_from_state(state, "truth-test-001")
        evidence = validator.extract_evidence_from_state(state, "truth-test-001")
        assert claim.receipt_hash == evidence.receipt_hash
        assert claim.receipt_hash != ""  # receipts must be non-empty

    def test_claim_tools_match_evidence_tool_calls(self):
        state = _make_full_state(tools=["search_web", "query_memory"])
        validator = ClaimEvidenceValidator()
        claim = validator.extract_claim_from_state(state, "truth-test-001")
        evidence = validator.extract_evidence_from_state(state, "truth-test-001")
        assert claim.tools_used == evidence.tool_calls

    def test_claim_memory_reads_match_evidence_events(self):
        state = _make_full_state(memory_keys=["key_a", "key_b", "key_c"])
        validator = ClaimEvidenceValidator()
        claim = validator.extract_claim_from_state(state, "truth-test-001")
        evidence = validator.extract_evidence_from_state(state, "truth-test-001")
        assert sorted(claim.memory_reads) == sorted(evidence.memory_events)

    def test_binding_result_has_correct_structure(self):
        state = _make_full_state()
        validator = ClaimEvidenceValidator()
        claim = validator.extract_claim_from_state(state, "truth-test-001")
        evidence = validator.extract_evidence_from_state(state, "truth-test-001")
        result = validator.validate(claim, evidence)
        assert isinstance(result, ClaimBindingResult)
        assert isinstance(result.valid, bool)
        assert isinstance(result.violations, list)
        assert result.claim is not None
        assert result.evidence is not None

    def test_to_dict_is_serialisable(self):
        import json
        state = _make_full_state()
        validator = ClaimEvidenceValidator()
        claim = validator.extract_claim_from_state(state, "truth-test-001")
        evidence = validator.extract_evidence_from_state(state, "truth-test-001")
        result = validator.validate(claim, evidence)
        d = result.to_dict()
        # Must be JSON-serialisable
        json_str = json.dumps(d)
        assert len(json_str) > 0


# ---------------------------------------------------------------------------
# Level 2: Mutation — deliberate divergence must produce violations
# ---------------------------------------------------------------------------

class TestTruthBindingMutationDetection:
    """Mutating one layer while keeping the other fixed must produce violations.

    This proves the validator is not just rubber-stamping claims.
    """

    def _base_claim_and_evidence(self) -> tuple[ExecutionClaim, ExecutionEvidence, str]:
        state = _make_full_state()
        validator = ClaimEvidenceValidator()
        claim = validator.extract_claim_from_state(state, "truth-test-001")
        evidence = validator.extract_evidence_from_state(state, "truth-test-001")
        return claim, evidence, "truth-test-001"

    def _validate(self, claim: ExecutionClaim, evidence: ExecutionEvidence) -> ClaimBindingResult:
        return ClaimEvidenceValidator().validate(claim, evidence)

    def test_mismatched_steps_produces_violation(self):
        claim, evidence, _ = self._base_claim_and_evidence()
        # Tamper with claim steps
        tampered_claim = ExecutionClaim(
            turn_id=claim.turn_id,
            steps=["plan", "EXTRA_GHOST_STEP", "respond"],  # claim says extra step
            tools_used=claim.tools_used,
            memory_reads=claim.memory_reads,
            receipt_hash=claim.receipt_hash,
        )
        result = self._validate(tampered_claim, evidence)
        assert not result.valid
        fields = {v.field for v in result.violations}
        assert "steps" in fields

    def test_mismatched_tools_produces_violation(self):
        claim, evidence, _ = self._base_claim_and_evidence()
        tampered_claim = ExecutionClaim(
            turn_id=claim.turn_id,
            steps=claim.steps,
            tools_used=["ghost_tool_never_ran"],  # claimed tool that didn't run
            memory_reads=claim.memory_reads,
            receipt_hash=claim.receipt_hash,
        )
        result = self._validate(tampered_claim, evidence)
        assert not result.valid
        fields = {v.field for v in result.violations}
        assert "tools_used" in fields

    def test_mismatched_memory_reads_produces_violation(self):
        claim, evidence, _ = self._base_claim_and_evidence()
        tampered_claim = ExecutionClaim(
            turn_id=claim.turn_id,
            steps=claim.steps,
            tools_used=claim.tools_used,
            memory_reads=["ghost_memory_key"],  # claimed memory read that didn't happen
            receipt_hash=claim.receipt_hash,
        )
        result = self._validate(tampered_claim, evidence)
        assert not result.valid
        fields = {v.field for v in result.violations}
        assert "memory_reads" in fields

    def test_mismatched_receipt_hash_produces_violation(self):
        claim, evidence, _ = self._base_claim_and_evidence()
        tampered_claim = ExecutionClaim(
            turn_id=claim.turn_id,
            steps=claim.steps,
            tools_used=claim.tools_used,
            memory_reads=claim.memory_reads,
            receipt_hash="FAKE_HASH_DOES_NOT_MATCH",
        )
        result = self._validate(tampered_claim, evidence)
        assert not result.valid
        fields = {v.field for v in result.violations}
        assert "receipt_hash" in fields

    def test_mismatched_turn_id_produces_violation(self):
        claim, evidence, _ = self._base_claim_and_evidence()
        tampered_claim = ExecutionClaim(
            turn_id="WRONG_TURN_ID",
            steps=claim.steps,
            tools_used=claim.tools_used,
            memory_reads=claim.memory_reads,
            receipt_hash=claim.receipt_hash,
        )
        result = self._validate(tampered_claim, evidence)
        assert not result.valid
        fields = {v.field for v in result.violations}
        assert "turn_id" in fields

    def test_multiple_violations_all_reported(self):
        claim, evidence, _ = self._base_claim_and_evidence()
        # Tamper with both steps and tools
        tampered_claim = ExecutionClaim(
            turn_id=claim.turn_id,
            steps=["only", "two", "steps"],
            tools_used=["wrong_tool"],
            memory_reads=claim.memory_reads,
            receipt_hash=claim.receipt_hash,
        )
        result = self._validate(tampered_claim, evidence)
        assert not result.valid
        assert len(result.violations) >= 2

    def test_tampered_evidence_receipt_produces_violation(self):
        """Tamper with evidence (as if receipt chain was forged) — must be caught."""
        state = _make_full_state()
        validator = ClaimEvidenceValidator()
        claim = validator.extract_claim_from_state(state, "truth-test-001")
        # Build evidence with a fake receipt hash
        evidence = ExecutionEvidence(
            turn_id="truth-test-001",
            steps=claim.steps,
            tool_calls=claim.tools_used,
            memory_events=claim.memory_reads,
            receipt_hash="FORGED_RECEIPT_HASH",
        )
        result = validator.validate(claim, evidence)
        assert not result.valid
        fields = {v.field for v in result.violations}
        assert "receipt_hash" in fields

    def test_violation_contains_claimed_and_actual_values(self):
        claim, evidence, _ = self._base_claim_and_evidence()
        tampered_claim = ExecutionClaim(
            turn_id=claim.turn_id,
            steps=["ghost_step"],
            tools_used=claim.tools_used,
            memory_reads=claim.memory_reads,
            receipt_hash=claim.receipt_hash,
        )
        result = self._validate(tampered_claim, evidence)
        step_violations = [v for v in result.violations if v.field == "steps"]
        assert len(step_violations) == 1
        v = step_violations[0]
        assert v.claimed == ["ghost_step"]
        assert v.actual == evidence.steps


# ---------------------------------------------------------------------------
# Level 3: Structural — receipt chain integrity
# ---------------------------------------------------------------------------

class TestTruthBindingStructuralIntegrity:
    """Receipt chain must form a tamper-evident binding between claim and evidence."""

    def test_receipt_chain_hash_is_non_empty(self):
        state = _make_full_state()
        receipts = state["_execution_receipts"]
        h = compute_receipt_chain_hash(receipts)
        assert isinstance(h, str) and len(h) > 0

    def test_receipt_chain_hash_changes_when_stage_removed(self):
        state = _make_full_state()
        all_receipts = state["_execution_receipts"]
        full_hash = compute_receipt_chain_hash(all_receipts)
        truncated_hash = compute_receipt_chain_hash(all_receipts[:-1])
        assert full_hash != truncated_hash

    def test_receipt_chain_hash_changes_when_signature_tampered(self):
        state = _make_full_state()
        receipts = list(state["_execution_receipts"])
        original_hash = compute_receipt_chain_hash(receipts)
        # Tamper with first receipt's signature field
        tampered = dict(receipts[0])
        tampered["signature"] = "TAMPERED"
        tampered_receipts = [tampered] + receipts[1:]
        tampered_hash = compute_receipt_chain_hash(tampered_receipts)
        assert original_hash != tampered_hash

    def test_binding_is_hash_based_not_order_independent(self):
        """Receipt chain hash is order-sensitive: reordering breaks it."""
        state = _make_full_state()
        receipts = list(state["_execution_receipts"])
        original_hash = compute_receipt_chain_hash(receipts)
        reordered = list(reversed(receipts))
        reordered_hash = compute_receipt_chain_hash(reordered)
        assert original_hash != reordered_hash

    def test_empty_turn_produces_empty_receipt_hash(self):
        state = build_synthetic_state(
            turn_id="empty-turn",
            stages=[],
        )
        h = compute_receipt_chain_hash(state["_execution_receipts"])
        assert h == ""

    def test_binding_result_serialises_all_violation_fields(self):
        import json
        claim = ExecutionClaim(
            turn_id="t1",
            steps=["x"],
            tools_used=["t"],
            memory_reads=["m"],
            receipt_hash="abc",
        )
        evidence = ExecutionEvidence(
            turn_id="t1",
            steps=["y"],
            tool_calls=["t"],
            memory_events=["m"],
            receipt_hash="abc",
        )
        result = ClaimEvidenceValidator().validate(claim, evidence)
        d = result.to_dict()
        json_str = json.dumps(d)
        assert "steps" in json_str
        assert "claimed" in json_str
        assert "actual" in json_str


# ---------------------------------------------------------------------------
# Level 4: End-to-end scenario — planning task truth binding
# ---------------------------------------------------------------------------

class TestTruthBindingPlanningScenario:
    """Simulate a realistic planning turn and validate truth binding end-to-end.

    This mirrors the user's requested structure:
      result["uril_claims"]   → what system says happened
      result["execution_trace"] → what actually happened

    Runs entirely offline using synthetic state.
    """

    def _run_planning_turn(self, turn_id: str = "e2e-planning-001") -> dict[str, Any]:
        """Simulate a deterministic planning turn and return a result envelope."""
        state = build_synthetic_state(
            turn_id=turn_id,
            stages=["plan", "infer", "tool", "memory", "respond"],
            tools=["search_web"],
            memory_keys=["consolidated_memories", "life_patterns"],
        )
        validator = ClaimEvidenceValidator()
        claim = validator.extract_claim_from_state(state, turn_id)
        evidence = validator.extract_evidence_from_state(state, turn_id)
        result = validator.validate(claim, evidence)
        return {
            "turn_id": turn_id,
            "uril_claims": {
                "steps": claim.steps,
                "tools_used": claim.tools_used,
                "memory_reads": claim.memory_reads,
                "receipt_hash": claim.receipt_hash,
            },
            "execution_trace": {
                "steps": evidence.steps,
                "tool_calls": evidence.tool_calls,
                "memory_events": evidence.memory_events,
                "receipt_hash": evidence.receipt_hash,
            },
            "binding_result": result,
            "raw_state": state,
        }

    def test_planning_turn_claim_steps_equal_evidence_steps(self):
        result = self._run_planning_turn()
        assert result["uril_claims"]["steps"] == result["execution_trace"]["steps"]

    def test_planning_turn_tools_match(self):
        result = self._run_planning_turn()
        assert result["uril_claims"]["tools_used"] == result["execution_trace"]["tool_calls"]

    def test_planning_turn_memory_reads_match(self):
        result = self._run_planning_turn()
        assert sorted(result["uril_claims"]["memory_reads"]) == sorted(result["execution_trace"]["memory_events"])

    def test_planning_turn_receipt_hash_match(self):
        result = self._run_planning_turn()
        assert result["uril_claims"]["receipt_hash"] == result["execution_trace"]["receipt_hash"]

    def test_planning_turn_binding_valid(self):
        result = self._run_planning_turn()
        binding = result["binding_result"]
        assert binding.valid, f"Truth binding failed:\n{binding.to_dict()}"

    def test_two_identical_planning_turns_same_binding(self):
        """Two turns with the same input must produce identical binding results."""
        r1 = self._run_planning_turn("turn-001")
        r2 = self._run_planning_turn("turn-001")
        assert r1["uril_claims"] == r2["uril_claims"]
        assert r1["execution_trace"] == r2["execution_trace"]
        assert r1["binding_result"].valid == r2["binding_result"].valid

    def test_planning_turn_result_envelope_has_required_keys(self):
        result = self._run_planning_turn()
        assert "uril_claims" in result
        assert "execution_trace" in result
        assert "binding_result" in result
        # Claim layer keys
        assert "steps" in result["uril_claims"]
        assert "tools_used" in result["uril_claims"]
        assert "memory_reads" in result["uril_claims"]
        assert "receipt_hash" in result["uril_claims"]
        # Evidence layer keys
        assert "steps" in result["execution_trace"]
        assert "tool_calls" in result["execution_trace"]
        assert "memory_events" in result["execution_trace"]
        assert "receipt_hash" in result["execution_trace"]
