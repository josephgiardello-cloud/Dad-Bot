"""Phase 5 — Conflict Resolution Engine.

Resolves contradictions introduced by:
  - Fallback chains (primary tool says A; fallback tool says B)
  - Partial outputs (incomplete results from degraded execution)
  - Multi-tool execution (multiple tools return different answers to the same query)

5.1  TrustWeightedMerger
     Merges conflicting outputs by weighting each candidate by the
     ConfidenceVector aggregate of the tool that produced it.

5.2  ConflictDetector
     Detects contradictory outputs across a result set.  Uses structural
     comparison (type mismatch), semantic emptiness checks, and an optional
     content-divergence heuristic.

5.3  ResolutionPolicy
     Three policies for conflict resolution:
       - HIGHEST_CONFIDENCE_WINS  — pick the single highest-aggregate result
       - ENSEMBLE_MERGE           — weighted merge of all non-empty results
       - DEFER_TO_REEXECUTION     — signal that re-execution is required
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from dadbot.core.uncertainty_model import ConfidenceVector, FusionStrategy, ConfidenceFusion

# ---------------------------------------------------------------------------
# Domain types
# ---------------------------------------------------------------------------


class ResolutionPolicyKind(str, Enum):
    HIGHEST_CONFIDENCE_WINS = "highest_confidence_wins"
    ENSEMBLE_MERGE = "ensemble_merge"
    DEFER_TO_REEXECUTION = "defer_to_reexecution"


class ConflictKind(str, Enum):
    TYPE_MISMATCH = "type_mismatch"          # outputs have different Python types
    EMPTY_VS_NONEMPTY = "empty_vs_nonempty"  # one output is empty, another is not
    VALUE_DIVERGENCE = "value_divergence"    # same type but content diverges
    STATUS_MISMATCH = "status_mismatch"      # different terminal statuses
    NONE = "none"                            # no conflict detected


@dataclass
class ToolOutput:
    """One candidate output from a single tool execution."""
    tool_name: str
    output: Any
    confidence_vector: ConfidenceVector
    status: str = "ok"
    is_partial: bool = False
    raw_error: str = ""

    def is_empty(self) -> bool:
        if self.output is None:
            return True
        if isinstance(self.output, (str, list, dict, bytes)) and len(self.output) == 0:
            return True
        return False

    def output_type_name(self) -> str:
        if self.output is None:
            return "null"
        return type(self.output).__name__


# ---------------------------------------------------------------------------
# 5.2 — Conflict Detector
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConflictReport:
    conflict_kind: ConflictKind
    conflicting_tools: list[str]
    description: str
    severity: float                  # 0.0 (benign) – 1.0 (critical)

    def has_conflict(self) -> bool:
        return self.conflict_kind != ConflictKind.NONE


class ConflictDetector:
    """Detects contradictions across a set of ToolOutputs.

    Detection logic (applied in order of severity):
      1. STATUS_MISMATCH  — outputs differ in terminal status
      2. TYPE_MISMATCH    — outputs have different Python types
      3. EMPTY_VS_NONEMPTY — some outputs are empty, others are not
      4. VALUE_DIVERGENCE  — same type but content fingerprints differ
    """

    # Content divergence threshold: if fingerprint equality rate < this, flag
    DIVERGENCE_THRESHOLD: float = 0.5

    def detect(self, outputs: list[ToolOutput]) -> ConflictReport:
        if len(outputs) < 2:
            return ConflictReport(
                conflict_kind=ConflictKind.NONE,
                conflicting_tools=[o.tool_name for o in outputs],
                description="Single output; no conflict possible.",
                severity=0.0,
            )

        # 1. Status mismatch
        statuses = {o.status for o in outputs}
        if len(statuses) > 1:
            return ConflictReport(
                conflict_kind=ConflictKind.STATUS_MISMATCH,
                conflicting_tools=[o.tool_name for o in outputs],
                description=f"Terminal statuses diverge: {sorted(statuses)}.",
                severity=0.7,
            )

        # 2. Type mismatch
        types = {o.output_type_name() for o in outputs if not o.is_empty()}
        if len(types) > 1:
            return ConflictReport(
                conflict_kind=ConflictKind.TYPE_MISMATCH,
                conflicting_tools=[o.tool_name for o in outputs],
                description=f"Output types diverge: {sorted(types)}.",
                severity=0.85,
            )

        # 3. Empty vs non-empty
        empty_tools = [o.tool_name for o in outputs if o.is_empty()]
        nonempty_tools = [o.tool_name for o in outputs if not o.is_empty()]
        if empty_tools and nonempty_tools:
            return ConflictReport(
                conflict_kind=ConflictKind.EMPTY_VS_NONEMPTY,
                conflicting_tools=empty_tools + nonempty_tools,
                description=f"Empty outputs from {empty_tools}; non-empty from {nonempty_tools}.",
                severity=0.5,
            )

        # 4. Value divergence — fingerprint-based
        fingerprints = [self._fingerprint(o.output) for o in outputs if not o.is_empty()]
        if len(set(fingerprints)) > 1:
            unique_ratio = len(set(fingerprints)) / len(fingerprints)
            if unique_ratio > self.DIVERGENCE_THRESHOLD:
                return ConflictReport(
                    conflict_kind=ConflictKind.VALUE_DIVERGENCE,
                    conflicting_tools=[o.tool_name for o in outputs if not o.is_empty()],
                    description=f"Content fingerprints diverge (unique ratio={unique_ratio:.2f}).",
                    severity=0.6,
                )

        return ConflictReport(
            conflict_kind=ConflictKind.NONE,
            conflicting_tools=[o.tool_name for o in outputs],
            description="No conflict detected.",
            severity=0.0,
        )

    @staticmethod
    def _fingerprint(value: Any) -> str:
        try:
            serialized = json.dumps(value, sort_keys=True, default=str)
        except Exception:
            serialized = str(value)
        return hashlib.sha1(serialized.encode("utf-8")).hexdigest()[:12]


# ---------------------------------------------------------------------------
# 5.1 + 5.3 — Trust-Weighted Merger + Resolution Policies
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResolutionResult:
    """The output of a conflict resolution pass."""
    resolved_output: Any
    winning_tool: str                    # tool whose output was selected or primary
    confidence_vector: ConfidenceVector  # confidence of the resolved output
    policy_used: ResolutionPolicyKind
    conflict_report: ConflictReport
    resolution_notes: list[str]
    requires_reexecution: bool = False


class TrustWeightedMerger:
    """Merges a list of ToolOutputs weighted by their ConfidenceVector.aggregate.

    Only works meaningfully for list and dict outputs.  For scalar/str outputs,
    falls back to HIGHEST_CONFIDENCE_WINS.
    """

    def merge(self, outputs: list[ToolOutput]) -> tuple[Any, ConfidenceVector]:
        """Merge non-empty outputs into a single result, weighted by confidence.

        Returns (merged_output, fused_confidence_vector).
        """
        nonempty = [o for o in outputs if not o.is_empty()]
        if not nonempty:
            # All empty — return empty list with zero confidence
            zero_cv = ConfidenceVector(0.0, 0.0, 0.0, source_tool="merged", result_status="error")
            return [], zero_cv

        # Collect list items weighted by confidence
        scored: list[tuple[float, Any]] = []
        for o in nonempty:
            weight = max(0.0, o.confidence_vector.aggregate)
            if isinstance(o.output, list):
                for item in o.output:
                    scored.append((weight, item))
            elif isinstance(o.output, dict):
                scored.append((weight, o.output))
            else:
                scored.append((weight, o.output))

        # Deduplicate while preserving weighted order
        seen_fps: dict[str, float] = {}
        ordered: list[Any] = []
        for weight, item in sorted(scored, key=lambda x: x[0], reverse=True):
            fp = ConflictDetector._fingerprint(item)
            if fp not in seen_fps:
                seen_fps[fp] = weight
                ordered.append(item)

        fused = ConfidenceFusion.bayesian([o.confidence_vector for o in nonempty])
        fallback_cv = ConfidenceVector(0.5, 0.5, 0.5, source_tool="merged", result_status="ok")
        fused_cv = ConfidenceVector(
            reliability_score=fused.reliability_score if fused else fallback_cv.reliability_score,
            freshness_score=fused.freshness_score if fused else fallback_cv.freshness_score,
            completeness_score=fused.completeness_score if fused else fallback_cv.completeness_score,
            source_tool="merged",
            result_status="ok",
        )
        return ordered, fused_cv


class ConflictResolver:
    """Central conflict resolution engine.

    Applies the appropriate ResolutionPolicy based on the ConflictReport and
    the ConfidenceVectors of each candidate output.

    Default policy selection:
      - No conflict → USE first output as-is
      - STATUS_MISMATCH / TYPE_MISMATCH → HIGHEST_CONFIDENCE_WINS
      - EMPTY_VS_NONEMPTY → HIGHEST_CONFIDENCE_WINS (pick non-empty)
      - VALUE_DIVERGENCE, severity < 0.7 → ENSEMBLE_MERGE
      - VALUE_DIVERGENCE, severity ≥ 0.7 → DEFER_TO_REEXECUTION
    """

    def __init__(self, detector: ConflictDetector | None = None) -> None:
        self._detector = detector or ConflictDetector()
        self._merger = TrustWeightedMerger()

    def resolve(
        self,
        outputs: list[ToolOutput],
        policy_override: ResolutionPolicyKind | None = None,
    ) -> ResolutionResult:
        if not outputs:
            zero_cv = ConfidenceVector(0.0, 0.0, 0.0, source_tool="none", result_status="error")
            return ResolutionResult(
                resolved_output=None,
                winning_tool="none",
                confidence_vector=zero_cv,
                policy_used=ResolutionPolicyKind.HIGHEST_CONFIDENCE_WINS,
                conflict_report=ConflictReport(ConflictKind.NONE, [], "No outputs.", 0.0),
                resolution_notes=["No candidate outputs to resolve."],
                requires_reexecution=True,
            )

        report = self._detector.detect(outputs)
        policy = policy_override or self._select_policy(report)

        if policy == ResolutionPolicyKind.HIGHEST_CONFIDENCE_WINS:
            return self._highest_confidence_wins(outputs, report, policy)
        elif policy == ResolutionPolicyKind.ENSEMBLE_MERGE:
            return self._ensemble_merge(outputs, report, policy)
        else:  # DEFER_TO_REEXECUTION
            return self._defer(outputs, report)

    # ------------------------------------------------------------------
    # Policy implementations
    # ------------------------------------------------------------------

    def _highest_confidence_wins(
        self, outputs: list[ToolOutput], report: ConflictReport, policy: ResolutionPolicyKind
    ) -> ResolutionResult:
        # Prefer non-empty outputs
        nonempty = [o for o in outputs if not o.is_empty()]
        candidates = nonempty if nonempty else outputs
        winner = max(candidates, key=lambda o: o.confidence_vector.aggregate)
        notes = [
            f"Selected output from '{winner.tool_name}' "
            f"(confidence={winner.confidence_vector.aggregate:.3f}).",
        ]
        if report.has_conflict():
            notes.append(f"Conflict detected: {report.conflict_kind.value} — {report.description}")
        return ResolutionResult(
            resolved_output=winner.output,
            winning_tool=winner.tool_name,
            confidence_vector=winner.confidence_vector,
            policy_used=policy,
            conflict_report=report,
            resolution_notes=notes,
        )

    def _ensemble_merge(
        self, outputs: list[ToolOutput], report: ConflictReport, policy: ResolutionPolicyKind
    ) -> ResolutionResult:
        merged_output, fused_cv = self._merger.merge(outputs)
        tools = [o.tool_name for o in outputs if not o.is_empty()]
        notes = [
            f"Ensemble merge of {len(outputs)} outputs from: {', '.join(tools)}.",
            f"Fused confidence: {fused_cv.aggregate:.3f}.",
        ]
        if report.has_conflict():
            notes.append(f"Resolved conflict: {report.conflict_kind.value}.")
        return ResolutionResult(
            resolved_output=merged_output,
            winning_tool="ensemble",
            confidence_vector=fused_cv,
            policy_used=policy,
            conflict_report=report,
            resolution_notes=notes,
        )

    def _defer(
        self, outputs: list[ToolOutput], report: ConflictReport
    ) -> ResolutionResult:
        # Pick the best available output as a placeholder, but flag re-execution
        nonempty = [o for o in outputs if not o.is_empty()]
        fallback = max(nonempty, key=lambda o: o.confidence_vector.aggregate) if nonempty else outputs[0]
        notes = [
            f"Conflict severity={report.severity:.2f} exceeds resolution threshold.",
            f"Deferred to re-execution; best available from '{fallback.tool_name}' used as placeholder.",
        ]
        return ResolutionResult(
            resolved_output=fallback.output,
            winning_tool=fallback.tool_name,
            confidence_vector=fallback.confidence_vector,
            policy_used=ResolutionPolicyKind.DEFER_TO_REEXECUTION,
            conflict_report=report,
            resolution_notes=notes,
            requires_reexecution=True,
        )

    # ------------------------------------------------------------------
    # Policy selection heuristic
    # ------------------------------------------------------------------

    @staticmethod
    def _select_policy(report: ConflictReport) -> ResolutionPolicyKind:
        if not report.has_conflict():
            return ResolutionPolicyKind.HIGHEST_CONFIDENCE_WINS
        if report.conflict_kind in {ConflictKind.STATUS_MISMATCH, ConflictKind.TYPE_MISMATCH}:
            return ResolutionPolicyKind.HIGHEST_CONFIDENCE_WINS
        if report.conflict_kind == ConflictKind.EMPTY_VS_NONEMPTY:
            return ResolutionPolicyKind.HIGHEST_CONFIDENCE_WINS
        if report.conflict_kind == ConflictKind.VALUE_DIVERGENCE:
            return (
                ResolutionPolicyKind.DEFER_TO_REEXECUTION
                if report.severity >= 0.7
                else ResolutionPolicyKind.ENSEMBLE_MERGE
            )
        return ResolutionPolicyKind.HIGHEST_CONFIDENCE_WINS
