from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import streamlit as st

RUNTIME_REJECTION_SESSION_KEY = "dad_runtime_last_rejection"
RUNTIME_TURN_TIMELINE_SESSION_KEY = "dad_runtime_turn_timeline"
PRIMARY_VIEW_SESSION_KEY = "primary_view"
UI_MOOD_SESSION_KEY = "ui_mood"

EVENT_TO_UI_SIGNAL = {
    "user_message": "user_intent",
    "thinking_update": "reasoning_state",
    "decision_event": "policy_decision",
    "assistant_reply": "assistant_response",
    "photo_request": "effect_photo",
    "tts_request": "effect_tts",
    "assistant_attachment_added": "effect_attachment",
    "guardrail_rejection": "guardrail_block",
}

SIGNAL_PRIORITY = {
    "guardrail_block": 0,
    "policy_decision": 1,
    "reasoning_state": 2,
    "assistant_response": 3,
    "effect_photo": 4,
    "effect_tts": 5,
    "effect_attachment": 6,
    "user_intent": 7,
}


@dataclass(frozen=True)
class RuntimeGuardrailView:
    action: str
    rule: str
    layer: str
    reason: str
    message: str
    severity: str
    fix_hint: str


def runtime_rejection_payload(exc: Exception, *, action: str) -> dict:
    diagnostics = {}
    diagnostic_fn = getattr(exc, "diagnostics", None)
    if callable(diagnostic_fn):
        try:
            diagnostics = dict(diagnostic_fn() or {})
        except Exception:
            diagnostics = {}
    return {
        "action": str(action or "runtime_call"),
        "reason": str(
            diagnostics.get("reason")
            or "Dad Runtime blocked this action to protect system safety and contract integrity.",
        ),
        "rule": str(diagnostics.get("rule") or "unspecified_rule"),
        "layer": str(diagnostics.get("layer") or "runtime_boundary"),
        "message": str(
            diagnostics.get("message") or str(exc) or exc.__class__.__name__,
        ),
    }


def record_runtime_rejection(exc: Exception, *, action: str) -> None:
    is_capability_rejection = (
        callable(getattr(exc, "diagnostics", None))
        or hasattr(exc, "rule")
        or hasattr(exc, "layer")
        or "CapabilityViolation" in exc.__class__.__name__
    )
    if not is_capability_rejection:
        return
    st.session_state[RUNTIME_REJECTION_SESSION_KEY] = runtime_rejection_payload(
        exc,
        action=action,
    )


def clear_runtime_rejection() -> None:
    st.session_state.pop(RUNTIME_REJECTION_SESSION_KEY, None)


def runtime_rule_fix_hint(rule: str) -> str:
    mapping = {
        "contract_must_exist": "Use a capability that already exists, or add this API method to CAPABILITY_MANIFEST before calling it.",
        "api_method_must_be_classified": "Classify the UIRuntimeAPI method in CAPABILITY_MANIFEST and assign the correct group/effects.",
        "event_type_must_be_known": "Use a store-layer event from ALLOWED_STORE_EVENT_TYPES, not an internal runtime event.",
        "event_type_must_be_declared_for_capability": "Call the capability that owns this event type, or update the capability contract deliberately.",
        "side_effects_must_be_declared_for_capability": "Declare the observed side-effect label in the capability contract, or remove that side effect from this path.",
        "event_type_must_be_known_store_type": "Emit only known store event types and keep runtime/internal event namespaces out of UI ingress.",
        "external_event_type_must_be_ingress_allowed": "Ingress only user_message, assistant_reply, or assistant_attachment_added via gateway ingest_event.",
        "all_mutating_events_must_declare_side_effect_labels": "Attach _side_effects labels to emitted mutating events so validation can trace what happened.",
    }
    normalized = str(rule or "").strip().lower()
    return mapping.get(
        normalized,
        "Follow the capability contract for this action and keep UI calls on declared boundary methods.",
    )


def runtime_guardrail_severity(rule: str) -> str:
    normalized = str(rule or "").strip().lower()
    if normalized in {
        "external_event_type_must_be_ingress_allowed",
        "event_type_must_be_known",
        "event_type_must_be_known_store_type",
    }:
        return "critical"
    if normalized in {
        "event_type_must_be_declared_for_capability",
        "contract_must_exist",
        "api_method_must_be_classified",
    }:
        return "high"
    if normalized in {
        "side_effects_must_be_declared_for_capability",
        "all_mutating_events_must_declare_side_effect_labels",
    }:
        return "medium"
    return "low"


def runtime_guardrail_view_from_session() -> RuntimeGuardrailView | None:
    details = st.session_state.get(RUNTIME_REJECTION_SESSION_KEY)
    if not isinstance(details, dict) or not details:
        return None
    action = str(details.get("action") or "runtime_call")
    rule = str(details.get("rule") or "unspecified_rule")
    layer = str(details.get("layer") or "runtime_boundary")
    reason = str(details.get("reason") or "Dad Runtime blocked this action.")
    message = str(details.get("message") or "")
    return RuntimeGuardrailView(
        action=action,
        rule=rule,
        layer=layer,
        reason=reason,
        message=message,
        severity=runtime_guardrail_severity(rule),
        fix_hint=runtime_rule_fix_hint(rule),
    )


def runtime_turn_timeline() -> list[dict]:
    timeline = st.session_state.setdefault(RUNTIME_TURN_TIMELINE_SESSION_KEY, [])
    if not isinstance(timeline, list):
        timeline = []
        st.session_state[RUNTIME_TURN_TIMELINE_SESSION_KEY] = timeline
    return timeline


def record_turn_timeline_event(
    *,
    thread_id: str,
    event_type: str,
    summary: str,
    payload: dict | None = None,
    severity: str = "info",
) -> None:
    timeline = runtime_turn_timeline()
    timeline.append(
        {
            "timestamp": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "thread_id": str(thread_id or "default"),
            "event_type": str(event_type or "unknown"),
            "signal": EVENT_TO_UI_SIGNAL.get(
                str(event_type or "").strip().lower(),
                "runtime_state",
            ),
            "summary": str(summary or ""),
            "severity": str(severity or "info"),
            "payload": dict(payload or {}),
        },
    )
    if len(timeline) > 160:
        del timeline[:-160]


def active_primary_view(default: str = "chat") -> str:
    normalized_default = str(default or "chat").strip().lower() or "chat"
    value = (
        str(
            st.session_state.get(PRIMARY_VIEW_SESSION_KEY, normalized_default) or normalized_default,
        )
        .strip()
        .lower()
        or normalized_default
    )
    st.session_state[PRIMARY_VIEW_SESSION_KEY] = value
    return value


def set_primary_view(view: str) -> str:
    normalized = str(view or "chat").strip().lower() or "chat"
    st.session_state[PRIMARY_VIEW_SESSION_KEY] = normalized
    return normalized


def current_ui_mood(default: str = "neutral") -> str:
    normalized_default = str(default or "neutral").strip().lower() or "neutral"
    value = (
        str(
            st.session_state.get(UI_MOOD_SESSION_KEY, normalized_default) or normalized_default,
        )
        .strip()
        .lower()
        or normalized_default
    )
    st.session_state[UI_MOOD_SESSION_KEY] = value
    return value


def set_ui_mood(mood: str) -> str:
    normalized = str(mood or "neutral").strip().lower() or "neutral"
    st.session_state[UI_MOOD_SESSION_KEY] = normalized
    return normalized
