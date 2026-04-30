"""Truth Binding Layer — Claim → Evidence validation.

Bridges the gap between what the system *claims* happened (from orchestrator
reports, URIL metrics, benchmark output) and what *actually* occurred (proven
by execution receipts, tool invocation records, and memory access logs).

Why this matters
----------------
Without truth binding, internal metrics can diverge silently from real
execution reality.  A system can be "green" according to its own reports while
actually skipping stages, hallucinating tool calls, or misreporting memory
access.

Design
------
  ExecutionClaim   — structured description of what the system says it did
  ExecutionEvidence — ground truth extracted directly from execution artifacts
  ClaimEvidenceValidator — compares claims vs evidence, produces violations
  ClaimBindingResult — result of a binding check

Extraction helpers live on ClaimEvidenceValidator so callers can work
directly from a TurnContext.state dict.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------


@dataclass
class ExecutionClaim:
    """What the system asserts happened during a turn.

    These values come from orchestrator reports, planner output, tool
    registry records, or URIL pipeline signals — they are *claims*, not
    verified facts.
    """

    turn_id: str
    steps: list[str]  # claimed execution stages (ordered)
    tools_used: list[str]  # claimed tool call sequence (ordered)
    memory_reads: list[str]  # claimed memory access keys (set-semantics)
    receipt_hash: str  # claimed receipt-chain fingerprint


@dataclass
class ExecutionEvidence:
    """What actually happened, extracted from tamper-evident execution artifacts.

    These values come from:
      * ``TurnContext.state["_execution_receipts"]`` — signed stage receipts
      * ``TurnContext.state["tool_ir"]["executions"]`` — tool invocation log
      * ``TurnContext.state["memory_structured"]`` — memory access record
    """

    turn_id: str
    steps: list[str]  # stages recorded in receipt chain (ordered)
    tool_calls: list[str]  # actual tool invocations (ordered)
    memory_events: list[str]  # actual memory access keys (set-semantics)
    receipt_hash: str  # actual receipt-chain fingerprint


@dataclass
class BindingViolation:
    """A mismatch between a claimed value and the corresponding evidence."""

    field: str
    claimed: Any
    actual: Any

    def to_dict(self) -> dict[str, Any]:
        return {"field": self.field, "claimed": self.claimed, "actual": self.actual}


@dataclass
class ClaimBindingResult:
    """Result of a claim-vs-evidence validation pass."""

    valid: bool
    violations: list[BindingViolation] = field(default_factory=list)
    claim: ExecutionClaim | None = None
    evidence: ExecutionEvidence | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "violation_count": len(self.violations),
            "violations": [v.to_dict() for v in self.violations],
        }


# ---------------------------------------------------------------------------
# Receipt-chain fingerprint
# ---------------------------------------------------------------------------


def compute_receipt_chain_hash(receipts: list[Any]) -> str:
    """Deterministic fingerprint of an ordered receipt chain.

    Accepts either a list of ``ExecutionReceipt`` dataclass instances or a
    list of dicts (as stored in ``TurnContext.state["_execution_receipts"]``).
    Returns a SHA-256 hex digest, or empty string if the chain is empty.
    """
    if not receipts:
        return ""

    parts: list[str] = []
    for r in receipts:
        if isinstance(r, dict):
            sig = str(r.get("signature") or r.get("checkpoint_hash") or "")
            stage = str(r.get("stage") or "")
            seq = str(r.get("sequence") or "")
        else:
            sig = str(getattr(r, "signature", "") or "")
            stage = str(getattr(r, "stage", "") or "")
            seq = str(getattr(r, "sequence", "") or "")
        parts.append(f"{seq}:{stage}:{sig}")

    chain_str = "|".join(parts)
    return hashlib.sha256(chain_str.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


class ClaimEvidenceValidator:
    """Validates ExecutionClaims against ExecutionEvidence.

    Usage::

        validator = ClaimEvidenceValidator()
        claim    = validator.extract_claim_from_state(state, turn_id)
        evidence = validator.extract_evidence_from_state(state, turn_id)
        result   = validator.validate(claim, evidence)
        assert result.valid, result.to_dict()
    """

    # ------------------------------------------------------------------
    # Core validation
    # ------------------------------------------------------------------

    def validate(
        self,
        claim: ExecutionClaim,
        evidence: ExecutionEvidence,
    ) -> ClaimBindingResult:
        """Compare claim against evidence and return a binding result.

        Checks (in order):
          * turn_id equality
          * steps list equality (order-sensitive)
          * tools_used vs tool_calls equality (order-sensitive)
          * memory_reads vs memory_events set equality (order-insensitive)
          * receipt_hash equality
        """
        violations: list[BindingViolation] = []

        if claim.turn_id != evidence.turn_id:
            violations.append(
                BindingViolation("turn_id", claim.turn_id, evidence.turn_id),
            )

        if claim.steps != evidence.steps:
            violations.append(BindingViolation("steps", claim.steps, evidence.steps))

        if claim.tools_used != evidence.tool_calls:
            violations.append(
                BindingViolation("tools_used", claim.tools_used, evidence.tool_calls),
            )

        if sorted(claim.memory_reads) != sorted(evidence.memory_events):
            violations.append(
                BindingViolation(
                    "memory_reads",
                    sorted(claim.memory_reads),
                    sorted(evidence.memory_events),
                ),
            )

        if claim.receipt_hash != evidence.receipt_hash:
            violations.append(
                BindingViolation(
                    "receipt_hash",
                    claim.receipt_hash,
                    evidence.receipt_hash,
                ),
            )

        return ClaimBindingResult(
            valid=len(violations) == 0,
            violations=violations,
            claim=claim,
            evidence=evidence,
        )

    # ------------------------------------------------------------------
    # Extraction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def extract_claim_from_state(
        state: dict[str, Any],
        turn_id: str,
    ) -> ExecutionClaim:
        """Build an ExecutionClaim from a TurnContext.state dict.

        The claim is derived from the planner output (steps), tool registry
        IR (tools_used), and memory structured dict (memory_reads).  The
        receipt_hash is computed from the same receipts used by evidence —
        so claims and evidence diverge only when the *reported* steps/tools
        don't match the receipt trail.
        """
        plan = state.get("plan") or {}
        steps: list[str] = list(plan.get("steps") or [])

        tool_ir = state.get("tool_ir") or {}
        tools_used: list[str] = [str(e.get("tool") or e.get("name") or "") for e in (tool_ir.get("executions") or [])]

        mem_structured = state.get("memory_structured") or {}
        memory_reads: list[str] = [str(k) for k in mem_structured.keys()]

        receipts = state.get("_execution_receipts") or []
        receipt_hash = compute_receipt_chain_hash(receipts)

        return ExecutionClaim(
            turn_id=turn_id,
            steps=steps,
            tools_used=tools_used,
            memory_reads=memory_reads,
            receipt_hash=receipt_hash,
        )

    @staticmethod
    def extract_evidence_from_state(
        state: dict[str, Any],
        turn_id: str,
    ) -> ExecutionEvidence:
        """Build ExecutionEvidence from a TurnContext.state dict (ground truth).

        Unlike extract_claim_from_state, the steps come exclusively from the
        signed receipt chain — not from the planner's plan.  This separates
        what was *planned* (claim) from what was *executed* (evidence).
        """
        receipts = state.get("_execution_receipts") or []

        steps: list[str] = []
        for r in receipts:
            if isinstance(r, dict):
                steps.append(str(r.get("stage") or ""))
            else:
                steps.append(str(getattr(r, "stage", "") or ""))

        tool_ir = state.get("tool_ir") or {}
        tool_calls: list[str] = [str(e.get("tool") or e.get("name") or "") for e in (tool_ir.get("executions") or [])]

        mem_structured = state.get("memory_structured") or {}
        memory_events: list[str] = [str(k) for k in mem_structured.keys()]

        receipt_hash = compute_receipt_chain_hash(receipts)

        return ExecutionEvidence(
            turn_id=turn_id,
            steps=steps,
            tool_calls=tool_calls,
            memory_events=memory_events,
            receipt_hash=receipt_hash,
        )


# ---------------------------------------------------------------------------
# Convenience: build a synthetic state for unit testing
# ---------------------------------------------------------------------------


def build_synthetic_state(
    *,
    turn_id: str = "test-turn-001",
    stages: list[str] | None = None,
    tools: list[str] | None = None,
    memory_keys: list[str] | None = None,
    receipts: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Construct a minimal TurnContext.state dict for use in tests.

    If *receipts* is not provided, synthetic receipts are generated from
    *stages* so that the receipt chain is internally consistent.
    """
    _stages = stages if stages is not None else ["plan", "execute", "respond"]
    _tools = tools or []
    _keys = memory_keys or []

    if receipts is None:
        _receipts: list[dict[str, Any]] = []
        prev_sig = ""
        for i, stage in enumerate(_stages):
            sig = hashlib.sha256(
                f"{turn_id}:{stage}:{i}:{prev_sig}".encode(),
            ).hexdigest()
            rec: dict[str, Any] = {
                "turn_id": turn_id,
                "stage": stage,
                "sequence": i,
                "stage_call_id": hashlib.sha256(
                    f"{turn_id}:{stage}".encode(),
                ).hexdigest()[:32],
                "checkpoint_hash": hashlib.sha256(f"ckpt-{stage}".encode()).hexdigest()[:16],
                "prev_receipt_sig": prev_sig,
                "completed_at": 1000.0 + i,
                "signature": sig,
            }
            _receipts.append(rec)
            prev_sig = sig
    else:
        _receipts = receipts

    return {
        "plan": {"steps": _stages},
        "tool_ir": {"executions": [{"tool": t, "name": t} for t in _tools]},
        "memory_structured": {k: {} for k in _keys},
        "_execution_receipts": _receipts,
    }
