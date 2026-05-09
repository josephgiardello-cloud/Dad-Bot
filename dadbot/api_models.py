from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict, Field


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


class DadBotAPIModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class SubconsciousFragment(DadBotAPIModel):
    fragment_id: str = ""
    memory_id: str = ""
    similarity_score: float = Field(0.0, ge=0.0, le=1.0)
    checksum: str = ""
    sovereign_event_checksum: str = ""
    summary: str = ""
    category: str = ""
    mood: str = ""
    event: str = ""
    timestamp: str = ""
    source: str = "subconscious_reflex"


class IntegrityStatus(DadBotAPIModel):
    merkle_check_passed: bool = True
    reason: str = ""
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    checked_at: str = Field(default_factory=utc_now_iso)


class ThinkingStep(DadBotAPIModel):
    step_id: str = ""
    thought_trace: str = ""
    target_node: str = ""
    confidence_score: float = Field(0.0, ge=0.0, le=1.0)
    emitted_at: float = 0.0


class TrustMeter(DadBotAPIModel):
    trust_level: int = Field(50, ge=0, le=100)
    openness_level: int = Field(50, ge=0, le=100)
    trust_credit: float = Field(0.5, ge=0.0, le=1.0)
    alignment_score: float = Field(0.5, ge=0.0, le=1.0)
    label: str = "steady"


class DriftAlarm(DadBotAPIModel):
    active: bool = False
    current_risk_level: str = "low"
    predicted_drift_probability: float = Field(0.0, ge=0.0, le=1.0)
    likely_trigger_category: str = "unknown"
    confidence_score: float = Field(0.0, ge=0.0, le=1.0)
    recommended_intervention: str = ""
    intervention_justification: str = ""


class TurnEnvelope(DadBotAPIModel):
    session_id: str
    request_id: str
    tenant_id: str
    response_text: str
    subconscious_metadata: list[SubconsciousFragment] = Field(default_factory=list)
    integrity_status: IntegrityStatus = Field(default_factory=IntegrityStatus)
    trust_meter: TrustMeter = Field(default_factory=TrustMeter)
    drift_alarm: DriftAlarm = Field(default_factory=DriftAlarm)
    thinking_state: list[ThinkingStep] = Field(default_factory=list)
    active_model: str = ""
    should_end: bool = False
    status: str = "completed"
    metadata: dict[str, Any] = Field(default_factory=dict)


class PulseEnvelope(DadBotAPIModel):
    session_id: str
    tenant_id: str = ""
    event_type: str = "pulse.snapshot"
    status: str = "idle"
    trust_meter: TrustMeter = Field(default_factory=TrustMeter)
    drift_alarm: DriftAlarm = Field(default_factory=DriftAlarm)
    integrity_status: IntegrityStatus = Field(default_factory=IntegrityStatus)
    inference_intent: str = "Idle"
    current_thoughts: list[ThinkingStep] = Field(default_factory=list)
    cognition_stream: list[ThinkingStep] = Field(default_factory=list)
    subconscious_metadata: list[SubconsciousFragment] = Field(default_factory=list)
    reflection_summary: dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=utc_now_iso)


def _as_dict(value: Mapping[str, Any] | dict[str, Any] | None) -> dict[str, Any]:
    return dict(value or {})


def _build_thinking_steps(raw_steps: Any) -> list[ThinkingStep]:
    items: list[ThinkingStep] = []
    for raw in list(raw_steps or []):
        if not isinstance(raw, Mapping):
            continue
        items.append(
            ThinkingStep.model_validate(
                {
                    "step_id": str(raw.get("step_id") or raw.get("event_id") or ""),
                    "thought_trace": str(raw.get("thought_trace") or raw.get("message") or raw.get("summary") or ""),
                    "target_node": str(raw.get("target_node") or raw.get("node") or raw.get("stage") or ""),
                    "confidence_score": float(raw.get("confidence_score") or raw.get("confidence") or 0.0),
                    "emitted_at": float(raw.get("emitted_at") or raw.get("timestamp") or 0.0),
                }
            )
        )
    return items


def _build_subconscious_fragments(raw_fragments: Any, *, limit: int = 5) -> list[SubconsciousFragment]:
    fragments: list[SubconsciousFragment] = []
    for raw in list(raw_fragments or [])[: max(0, int(limit))]:
        if not isinstance(raw, Mapping):
            continue
        fragments.append(
            SubconsciousFragment.model_validate(
                {
                    "fragment_id": str(raw.get("fragment_id") or raw.get("memory_id") or raw.get("id") or raw.get("slug") or ""),
                    "memory_id": str(raw.get("memory_id") or raw.get("id") or raw.get("slug") or raw.get("key") or ""),
                    "similarity_score": float(raw.get("similarity_score") or 0.0),
                    "checksum": str(raw.get("checksum") or raw.get("content_hash") or raw.get("hash") or ""),
                    "sovereign_event_checksum": str(raw.get("sovereign_event_checksum") or raw.get("signed_sovereign_checksum") or ""),
                    "summary": str(raw.get("summary") or raw.get("text") or ""),
                    "category": str(raw.get("category") or ""),
                    "mood": str(raw.get("mood") or ""),
                    "event": str(raw.get("event") or ""),
                    "timestamp": str(raw.get("timestamp") or raw.get("created_at") or raw.get("updated_at") or ""),
                    "source": str(raw.get("source") or "subconscious_reflex"),
                }
            )
        )
    return fragments


def _build_integrity_status(session_state: Mapping[str, Any] | None, session_metadata: Mapping[str, Any] | None) -> IntegrityStatus:
    state = _as_dict(session_state)
    metadata = _as_dict(session_metadata)
    last_integrity_status = _as_dict(state.get("last_integrity_status"))
    integrity_failure = bool(metadata.get("integrity_failure") or state.get("integrity_failure"))
    if not integrity_failure:
        integrity_failure = not bool(last_integrity_status.get("merkle_check_passed", True))
    return IntegrityStatus.model_validate(
        {
            "merkle_check_passed": not integrity_failure,
            "reason": str(
                metadata.get("integrity_failure_reason")
                or state.get("integrity_failure_reason")
                or last_integrity_status.get("reason")
                or ""
            ),
            "diagnostics": dict(
                metadata.get("integrity_failure_diagnostics")
                or state.get("integrity_failure_diagnostics")
                or last_integrity_status.get("diagnostics")
                or {}
            ),
        }
    )


def _build_trust_meter(session_state: Mapping[str, Any] | None) -> TrustMeter:
    relationship = _as_dict(session_state.get("relationship_state") if isinstance(session_state, Mapping) else {})
    trust_level = int(relationship.get("trust_level") or 50)
    openness_level = int(relationship.get("openness_level") or 50)
    trust_credit = max(0.0, min(1.0, trust_level / 100.0))
    alignment_score = max(0.0, min(1.0, (trust_level + openness_level) / 200.0))
    return TrustMeter.model_validate(
        {
            "trust_level": trust_level,
            "openness_level": openness_level,
            "trust_credit": trust_credit,
            "alignment_score": alignment_score,
            "label": str(relationship.get("trust_label") or relationship.get("emotional_momentum") or "steady"),
        }
    )


def _build_drift_alarm(session_state: Mapping[str, Any] | None) -> DriftAlarm:
    reflection = _as_dict(
        session_state.get("last_reflection_summary") if isinstance(session_state, Mapping) else {}
    )
    risk_level = str(reflection.get("current_risk_level") or "low").strip().lower()
    probability = float(reflection.get("predicted_drift_probability") or 0.0)
    confidence = float(reflection.get("confidence_score") or 0.0)
    return DriftAlarm.model_validate(
        {
            "active": risk_level in {"high", "critical"} or probability >= 0.5,
            "current_risk_level": risk_level,
            "predicted_drift_probability": probability,
            "likely_trigger_category": str(reflection.get("likely_trigger_category") or "unknown"),
            "confidence_score": confidence,
            "recommended_intervention": str(reflection.get("recommended_intervention") or ""),
            "intervention_justification": str(reflection.get("intervention_justification") or ""),
        }
    )


def build_turn_envelope(
    *,
    session_id: str,
    request_id: str,
    tenant_id: str,
    response_payload: Mapping[str, Any] | None,
    session_state: Mapping[str, Any] | None,
    session_metadata: Mapping[str, Any] | None,
) -> TurnEnvelope:
    state = _as_dict(session_state)
    response = _as_dict(response_payload)
    metadata = _as_dict(session_metadata)
    return TurnEnvelope.model_validate(
        {
            "session_id": str(session_id or "default"),
            "request_id": str(request_id or ""),
            "tenant_id": str(tenant_id or "default"),
            "response_text": str(response.get("reply") or response.get("response_text") or ""),
            "subconscious_metadata": [fragment.model_dump(mode="json") for fragment in _build_subconscious_fragments(state.get("memory_retrieval_set") or state.get("subconscious_memory_fragments") or [], limit=5)],
            "integrity_status": _build_integrity_status(state, metadata).model_dump(mode="json"),
            "trust_meter": _build_trust_meter(state).model_dump(mode="json"),
            "drift_alarm": _build_drift_alarm(state).model_dump(mode="json"),
            "thinking_state": [step.model_dump(mode="json") for step in _build_thinking_steps(state.get("cognition_stream") or [])],
            "active_model": str(response.get("active_model") or ""),
            "should_end": bool(response.get("should_end", state.get("should_end", False))),
            "status": str(response.get("status") or "completed"),
            "metadata": dict(response.get("metadata") or {}),
        }
    )


def build_pulse_envelope(
    *,
    session_id: str,
    tenant_id: str,
    session_state: Mapping[str, Any] | None,
    session_metadata: Mapping[str, Any] | None,
    event_type: str = "pulse.snapshot",
) -> PulseEnvelope:
    state = _as_dict(session_state)
    metadata = _as_dict(session_metadata)
    cognition_stream = _build_thinking_steps(state.get("cognition_stream") or [])
    reflection = _as_dict(state.get("last_reflection_summary") or state.get("reflection_summary") or {})
    current_thoughts = cognition_stream[-3:]
    inference_intent = current_thoughts[-1].thought_trace if current_thoughts else "Idle"
    if not inference_intent or inference_intent == "":
        inference_intent = "Idle"
    terminal_state = _as_dict(state.get("last_terminal_state"))
    status = str(terminal_state.get("status") or terminal_state.get("state") or "").strip().lower()
    if not status:
        status = "thinking" if cognition_stream else "idle"
    if terminal_state and inference_intent == "Idle":
        inference_intent = "Ready"
    return PulseEnvelope.model_validate(
        {
            "session_id": str(session_id or "default"),
            "tenant_id": str(tenant_id or ""),
            "event_type": str(event_type or "pulse.snapshot"),
            "status": status,
            "trust_meter": _build_trust_meter(state).model_dump(mode="json"),
            "drift_alarm": _build_drift_alarm(state).model_dump(mode="json"),
            "integrity_status": _build_integrity_status(state, metadata).model_dump(mode="json"),
            "inference_intent": inference_intent,
            "current_thoughts": [step.model_dump(mode="json") for step in current_thoughts],
            "cognition_stream": [step.model_dump(mode="json") for step in cognition_stream],
            "subconscious_metadata": [fragment.model_dump(mode="json") for fragment in _build_subconscious_fragments(state.get("subconscious_memory_fragments") or state.get("memory_retrieval_set") or [], limit=5)],
            "reflection_summary": dict(reflection),
        }
    )


__all__ = [
    "DadBotAPIModel",
    "DriftAlarm",
    "IntegrityStatus",
    "PulseEnvelope",
    "SubconsciousFragment",
    "ThinkingStep",
    "TrustMeter",
    "TurnEnvelope",
    "build_pulse_envelope",
    "build_turn_envelope",
]