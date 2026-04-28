from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from dadbot.models import BoundaryComplianceDeclaration, EvaluationContract, InvarianceGate

BoundaryName = Literal["boot", "registry", "orchestrator"]


def _stable_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


_DEFAULT_EVALUATION_CONTRACT = EvaluationContract(
    version="1.0",
    owner="dadbot",
    behavioral_invariance=InvarianceGate(
        role="CORE",
        question="Does the system behave identically?",
        signals=["replay_hash", "tool_trace_hash"],
        allowed_influences=[
            "canonical replay events",
            "tool execution trace",
            "semantic state transitions encoded by the replay ledger",
        ],
        must_not_depend_on=[
            "determinism_manifest_hash",
            "lock_hash",
            "instrumentation_mode",
            "PYTEST_CURRENT_TEST",
            "environment variable key-set changes",
        ],
        coverage=[
            "replay_hash captures orchestration graph and memory-transition effects reflected in the canonical replay ledger",
            "tool_trace_hash captures tool selection and tool-execution behavior",
        ],
        correctness_critical=True,
    ),
    envelope_invariance=InvarianceGate(
        role="SECONDARY",
        question="Is the execution context identical?",
        signals=["determinism_manifest_hash", "lock_hash"],
        examples=[
            "environment variables",
            "test/runtime flags",
            "instrumentation mode",
            "manifest metadata",
        ],
        correctness_critical=False,
    ),
)

_BOUNDARY_NOTES: dict[BoundaryName, tuple[str, ...]] = {
    "boot": (
        "Boot declares the canonical evaluation contract before profile and memory hydration.",
        "Boot must not redefine behavioral invariance from environment-only metadata.",
    ),
    "registry": (
        "Registry exposes services under the same evaluation contract hash used by runtime orchestration.",
        "Registry aliases and service lookup must not change behavioral gate semantics.",
    ),
    "orchestrator": (
        "Orchestrator enforces the evaluation contract at turn-execution boundaries.",
        "Orchestrator may record envelope drift, but overall pass depends only on behavioral invariance.",
    ),
}


def get_evaluation_contract() -> EvaluationContract:
    return _DEFAULT_EVALUATION_CONTRACT.model_copy(deep=True)


def evaluation_contract_payload() -> dict[str, Any]:
    return get_evaluation_contract().model_dump(mode="json")


def evaluation_contract_hash() -> str:
    return _stable_sha256(evaluation_contract_payload())


def build_boundary_compliance(
    boundary: BoundaryName,
    *,
    compliant: bool = True,
    notes: list[str] | None = None,
) -> BoundaryComplianceDeclaration:
    contract = get_evaluation_contract()
    merged_notes = list(_BOUNDARY_NOTES.get(boundary, ()))
    if notes:
        merged_notes.extend(str(note) for note in notes if str(note).strip())
    return BoundaryComplianceDeclaration(
        boundary=boundary,
        contract_version=contract.version,
        contract_hash=evaluation_contract_hash(),
        compliant=bool(compliant),
        declared_behavioral_signals=list(contract.behavioral_invariance.signals),
        declared_envelope_signals=list(contract.envelope_invariance.signals),
        notes=merged_notes,
    )


def resolve_boundary_declaration(
    boundary: BoundaryName,
    declaration: BoundaryComplianceDeclaration | dict[str, Any] | None,
) -> BoundaryComplianceDeclaration:
    if declaration is None:
        return build_boundary_compliance(
            boundary,
            compliant=False,
            notes=["missing boundary compliance declaration"],
        )

    model = BoundaryComplianceDeclaration.model_validate(declaration)
    notes = list(model.notes)
    compliant = bool(model.compliant)
    expected_hash = evaluation_contract_hash()
    expected_version = get_evaluation_contract().version

    if model.contract_hash != expected_hash:
        compliant = False
        notes.append("contract hash mismatch")
    if model.contract_version != expected_version:
        compliant = False
        notes.append("contract version mismatch")

    deduped_notes: list[str] = []
    seen: set[str] = set()
    for note in notes:
        normalized = str(note).strip()
        if normalized and normalized not in seen:
            deduped_notes.append(normalized)
            seen.add(normalized)

    return model.model_copy(update={"compliant": compliant, "notes": deduped_notes})


def serialize_boundary_declarations(
    declarations: dict[str, BoundaryComplianceDeclaration | dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    payload: dict[str, dict[str, Any]] = {}
    for name, declaration in declarations.items():
        model = BoundaryComplianceDeclaration.model_validate(declaration)
        payload[str(name)] = model.model_dump(mode="json")
    return payload


__all__ = [
    "BoundaryName",
    "build_boundary_compliance",
    "evaluation_contract_hash",
    "evaluation_contract_payload",
    "get_evaluation_contract",
    "resolve_boundary_declaration",
    "serialize_boundary_declarations",
]
