import os
import urllib.request as urllib_request
os.environ["DADBOT_VALIDATE_MANAGERS_SMOKE"] = "1"

# --- SYSTEMIC SINGLETON WIRING ---
from dadbot.core.services import build_services
from dadbot.core.dadbot import DadBot
from dadbot.config import DadBotConfig, DadRuntimeConfig
from dadbot.runtime_core.streamlit_runtime import StreamlitRuntime, UIRuntimeAPI
from dadbot.managers.profile_runtime import ProfileRuntimeManager
import dadbot.ui.interaction_controller as interaction_controller


import threading
import os
os.environ["DADBOT_VALIDATE_MANAGERS_SMOKE"] = "1"

# --- Explicit manager instantiation ---

# Manager imports and instantiations removed for missing/optional managers
memory_manager = None
relationship_manager = None
mood_manager = None
event_bus = None






runtime_config = DadRuntimeConfig()
config = DadBotConfig(runtime_config=runtime_config)
services = build_services()


from dadbot.registry import (
    wire_bootstrap_managers,
    wire_runtime_managers
)




# --- Create DadBot with only required arguments ---
REAL_DADBOT = DadBot(
    config=config,
    runtime_config=runtime_config,
    services=services,
    validate_managers=False,
)


# Wire all required managers after construction
from dadbot.registry import wire_bootstrap_managers, wire_runtime_managers
wire_bootstrap_managers(REAL_DADBOT)
wire_runtime_managers(REAL_DADBOT)

# Optional: verify wiring worked
print("Managers wired successfully. Available managers:",
      [attr for attr in dir(REAL_DADBOT) if not attr.startswith('_') and 'manager' in attr.lower()])

# Optional: run strict validation after wiring
# try:
#     REAL_DADBOT.services.validate_facade(smoke=False)
#     print("✅ DadBot strict facade validation passed")
# except Exception as e:
#     print("❌ Strict validation failed:", e)


## Load state at startup (disabled: function not defined)
# load_bot_state(REAL_DADBOT)

REAL_DADBOT.STYLE = {"name": "DadBot"}
REAL_DADBOT._session_lock = threading.RLock()
REAL_DADBOT.thread_snapshots = {}


# Direct, non-cached, non-patched runtime and API
UI_RUNTIME = StreamlitRuntime(REAL_DADBOT)
UI_API = UIRuntimeAPI(REAL_DADBOT)

# Wire controller helpers to use these singletons
interaction_controller.get_runtime = lambda: UI_RUNTIME
interaction_controller.get_chat_event_api = lambda: UI_API
# --- END SYSTEMIC WIRING ---

import inspect
from dadbot.core.policy_store import DadPolicy, DadPolicyStore
from dadbot.core.runtime_service_provider import DefaultCoreRuntimeServices
# --- MISSING HELPER IMPLEMENTATIONS ---
def _resolve_policy_store(bot):
    """Resolve the DadPolicyStore from the bot instance."""
    try:
        if hasattr(bot, 'services') and hasattr(bot.services, 'get_policy_store'):
            store = bot.services.get_policy_store()
        else:
            store = DefaultCoreRuntimeServices(bot).get_policy_store()
        if isinstance(store, DadPolicyStore):
            return store
    except Exception:
        pass
    return None

def _run_sync_policy_call(func, *args, **kwargs):
    """Run an async or sync policy store method and return (result, error_dict)."""
    try:
        if inspect.iscoroutinefunction(func):
            result = asyncio.run(func(*args, **kwargs))
        else:
            result = func(*args, **kwargs)
        return result, None
    except Exception as exc:
        return None, {"detail": str(exc)}

def _sidebar_policy_store_health(bot):
    """Return a dict with policy store health info."""
    store = _resolve_policy_store(bot)
    if not store:
        return {"status": "unavailable", "detail": "Policy store not available"}
    try:
        policy, err = _run_sync_policy_call(store.get_current_policy)
        if err:
            return {"status": "unavailable", "detail": err.get("detail", "Error loading policy")}
        return {"status": "healthy", "version": getattr(policy, 'version', 'unknown'), "detail": "policy store healthy"}
    except Exception as exc:
        return {"status": "unavailable", "detail": str(exc)}

# --- MINIMAL STATUS TAB ---
def render_status_tab(bot):
    st.subheader("System Status")
    policy_health = _sidebar_policy_store_health(bot)
    st.markdown(f"**Policy Store:** {policy_health.get('status', 'unknown').capitalize()} - {policy_health.get('detail', '')}")
    st.markdown(f"**Policy Version:** {policy_health.get('version', 'unknown')}")
    st.caption("Add more diagnostic panels here as needed.")
import sys
import asyncio
import base64
import hashlib
import importlib
import io
import json
from urllib.parse import urlencode
import logging
import mimetypes
import os
import re
import shutil
import subprocess
import tempfile
import time
import uuid
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from urllib import error as urllib_error
## (removed duplicate import os and env var set)
os.environ["DADBOT_VALIDATE_MANAGERS_SMOKE"] = "1"

try:
    import streamlit as st  # pyright: ignore[reportMissingImports]
    import streamlit.components.v1 as components  # pyright: ignore[reportMissingImports]
except Exception:  # pragma: no cover - optional dependency
    st = cast(Any, None)
    components = cast(Any, None)

try:
    from PIL import Image, ImageDraw, ImageFont  # pyright: ignore[reportMissingImports]
except Exception:  # pragma: no cover - optional dependency
    Image = cast(Any, None)
    ImageDraw = cast(Any, None)
    ImageFont = cast(Any, None)

from dadbot.consumers.streamlit import load_thread_projection
from dadbot.core.policy_store import DadPolicy, DadPolicyStore
from dadbot.runtime.supervisor import get_runtime_supervisor
from dadbot.runtime_core import ThreadView, UIRuntimeAPI
from dadbot.runtime_core.streamlit_runtime import StreamlitRuntime
from dadbot.streamlit_helpers import (
    format_relationship_card,
    get_relationship_health_stats,
    load_important_memories,
    save_important_memory,
)
from dadbot.ui import interaction_controller, state_manager
from dadbot.ui.data import render_data_tab
from dadbot.ui.helpers import (
    apply_power_mode,
    apply_ui_preferences,
    find_available_image_model,
    local_stt_backend_status,
    local_tts_backend_status,
    render_voice_dependency_help,
)
from dadbot.ui.preferences import render_preferences_tab
from dadbot.ui.prefs_state import (
    sync_ui_voice_from_profile,
    ui_preferences,
    voice_preferences,
)
from dadbot.ui.status import render_confluence_status_card
from dadbot.ui.utils import (
    ambient_fragment,
    maybe_fragment,
    titleize_token,
)
from dadbot.ui.voice_control_plane import VoiceSessionController

if TYPE_CHECKING:
    from dadbot.core.dadbot import DadBot

try:
    STAGE_METADATA = importlib.import_module("dadbot.core.system_behavior_views").STAGE_METADATA
except Exception:  # pragma: no cover - optional compatibility shim
    STAGE_METADATA = {}

logger = logging.getLogger(__name__)

RUNTIME_REJECTION_SESSION_KEY = state_manager.RUNTIME_REJECTION_SESSION_KEY
RUNTIME_TURN_TIMELINE_SESSION_KEY = state_manager.RUNTIME_TURN_TIMELINE_SESSION_KEY
EVENT_TO_UI_SIGNAL = state_manager.EVENT_TO_UI_SIGNAL
SIGNAL_PRIORITY = state_manager.SIGNAL_PRIORITY

try:
    from faster_whisper import WhisperModel  # pyright: ignore[reportMissingImports]
except Exception:  # pragma: no cover - optional dependency
    WhisperModel = None

try:
    import whisper as openai_whisper  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - optional dependency
    openai_whisper = None

try:
    import pyttsx3  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - optional dependency
    pyttsx3 = None

try:
    from streamlit_webrtc import RTCConfiguration as WebRtcRTCConfiguration  # pyright: ignore[reportMissingImports]
    from streamlit_webrtc import WebRtcMode, webrtc_streamer  # pyright: ignore[reportMissingImports]

    _WEBRTC_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    _WEBRTC_AVAILABLE = False
    WebRtcRTCConfiguration = cast(Any, None)
    WebRtcMode = cast(Any, None)
    webrtc_streamer = cast(Any, None)

STATIC_DIR = Path("static")
DAD_AVATAR_PATH = STATIC_DIR / "dad_avatar.png"
STATIC_DIR.mkdir(exist_ok=True)

INITIAL_GREETING = "That's my boy. I love hearing that, Tony."
DEFAULT_EXPORT_PATH = Path("memory_export.json")
RUNTIME_OUTBOX_PATH = Path("session_logs/runtime_outbox.json")
PWA_MANIFEST_FILE = "dadbot-manifest.webmanifest"
PWA_ICON_FILE = "dadbot-icon.svg"
PWA_MASKABLE_ICON_FILE = "dadbot-icon-maskable.svg"
PWA_HEAD_SYNC_FILE = "dadbot-head-sync.html"
UI_STYLESHEET_FILE = "dadbot-ui.css"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
CHAT_EVENT_JOURNAL_PATH = Path("runtime") / "chat_runtime_events.jsonl"
CONTEXTUAL_LEARNING_KEYWORDS: tuple[str, ...] = (
    "family",
    "mom",
    "mother",
    "dad",
    "father",
    "sister",
    "brother",
    "wife",
    "husband",
    "son",
    "daughter",
    "grandma",
    "grandpa",
    "health",
    "hospital",
    "diagnosis",
    "medication",
    "surgery",
    "therapy",
    "illness",
    "life event",
    "birthday",
    "wedding",
    "anniversary",
    "graduation",
    "funeral",
    "new job",
    "promotion",
    "moved",
    "moving",
    "divorce",
)
STORY_MODE_MAX_FAILED_ATTEMPTS = 3
STORY_MODE_LOCKOUT_BASE_SECONDS = 30
STORY_MODE_LOCKOUT_MAX_SECONDS = 900
WORLD_MODEL_SESSION_KEY = "world_model"
WORLD_MODEL_HISTORY_SESSION_KEY = "world_model_history"


def configured_story_mode_password() -> str:
    return str(os.environ.get("DADBOT_STORY_MODE_PASSWORD", "")).strip()


def story_mode_lockout_remaining_seconds() -> int:
    lock_until = float(st.session_state.get("chat_story_mode_lock_until_epoch", 0.0) or 0.0)
    remaining = lock_until - time.time()
    return max(0, int(remaining))


def reset_story_mode_password_failures() -> None:
    st.session_state["chat_story_password_failed_attempts"] = 0
    st.session_state["chat_story_mode_lock_until_epoch"] = 0.0


def register_story_mode_password_failure() -> int:
    failed_attempts = int(st.session_state.get("chat_story_password_failed_attempts", 0) or 0) + 1
    st.session_state["chat_story_password_failed_attempts"] = failed_attempts
    if failed_attempts < STORY_MODE_MAX_FAILED_ATTEMPTS:
        return 0

    lockout_level = failed_attempts - STORY_MODE_MAX_FAILED_ATTEMPTS
    lockout_seconds = STORY_MODE_LOCKOUT_BASE_SECONDS * (2**lockout_level)
    lockout_seconds = min(lockout_seconds, STORY_MODE_LOCKOUT_MAX_SECONDS)
    st.session_state["chat_story_mode_lock_until_epoch"] = time.time() + float(lockout_seconds)
    return int(lockout_seconds)


def should_trigger_contextual_learning(prompt: str) -> bool:
    normalized = str(prompt or "").strip().lower()
    if not normalized:
        return False
    return any(keyword in normalized for keyword in CONTEXTUAL_LEARNING_KEYWORDS)


def pwa_asset_url(file_name):
    return f"/app/static/{file_name}"


def pwa_asset_status():
    return {
        "manifest": (STATIC_DIR / PWA_MANIFEST_FILE).exists(),
        "icon": (STATIC_DIR / PWA_ICON_FILE).exists(),
        "maskable_icon": (STATIC_DIR / PWA_MASKABLE_ICON_FILE).exists(),
    }


@st.cache_data(show_spinner=False)
def load_ui_stylesheet() -> str:
    css_path = STATIC_DIR / UI_STYLESHEET_FILE
    try:
        return css_path.read_text(encoding="utf-8")
    except Exception as exc:
        logger.warning("Failed to read UI stylesheet %s: %s", css_path, exc)
        return ""


def _runtime_rejection_payload(exc: Exception, *, action: str) -> dict:
    return state_manager.runtime_rejection_payload(exc, action=action)


def record_runtime_rejection(exc: Exception, *, action: str) -> None:
    state_manager.record_runtime_rejection(exc, action=action)


def render_runtime_rejection_banner(*, dismiss_key: str = "dismiss-runtime-rejection") -> None:
    details = st.session_state.get(RUNTIME_REJECTION_SESSION_KEY)
    if not isinstance(details, dict) or not details:
        return
    with st.container(border=True):
        st.error("Dad Runtime blocked a capability request.")
        st.caption(f"Action: {details.get('action', 'runtime_call')}")
        st.caption(f"Why blocked: {details.get('reason', 'Unknown reason')}")
        st.caption(f"Rule triggered: {details.get('rule', 'unspecified_rule')}")
        st.caption(f"Rejected by layer: {details.get('layer', 'runtime_boundary')}")
        st.caption(f"Details: {details.get('message', '')}")
        if st.button("Dismiss", key=dismiss_key, use_container_width=False):
            state_manager.clear_runtime_rejection()
            st.rerun()


def runtime_rule_fix_hint(rule: str) -> str:
    return state_manager.runtime_rule_fix_hint(rule)


RuntimeGuardrailView = state_manager.RuntimeGuardrailView


def runtime_guardrail_severity(rule: str) -> str:
    return state_manager.runtime_guardrail_severity(rule)


def runtime_guardrail_view_from_session() -> RuntimeGuardrailView | None:
    return state_manager.runtime_guardrail_view_from_session()


def runtime_turn_timeline() -> list[dict]:
    return state_manager.runtime_turn_timeline()


def record_turn_timeline_event(
    *, thread_id: str, event_type: str, summary: str, payload: dict | None = None, severity: str = "info"
) -> None:
    state_manager.record_turn_timeline_event(
        thread_id=thread_id,
        event_type=event_type,
        summary=summary,
        payload=payload,
        severity=severity,
    )


def runtime_semantic_signal_snapshot(*, view: ThreadView | None, guardrail: RuntimeGuardrailView | None) -> list[dict]:
    signals: list[dict] = []
    seen: set[str] = set()

    turn_graph = dict((view.turn_graph if view else {}) or {})
    event_sequence = [str(item or "").strip().lower() for item in list(turn_graph.get("event_sequence") or [])]
    for event_name in event_sequence:
        signal_name = EVENT_TO_UI_SIGNAL.get(event_name)
        if signal_name and signal_name not in seen:
            seen.add(signal_name)
            signals.append({"signal": signal_name, "source": event_name, "severity": "info"})

    thinking = dict((view.thinking if view else {}) or {})
    if thinking and "reasoning_state" not in seen:
        seen.add("reasoning_state")
        signals.append(
            {
                "signal": "reasoning_state",
                "source": "thinking_update",
                "severity": "info",
                "summary": str(thinking.get("final_path") or "model_reply"),
            }
        )

    decision = dict((view.decision if view else {}) or {})
    if decision and "policy_decision" not in seen:
        seen.add("policy_decision")
        signals.append(
            {
                "signal": "policy_decision",
                "source": "decision_event",
                "severity": "info",
                "summary": ", ".join(key for key, value in dict(decision.get("decisions") or {}).items() if bool(value))
                or "no side effects",
            }
        )

    if guardrail is not None and "guardrail_block" not in seen:
        signals.append(
            {
                "signal": "guardrail_block",
                "source": "capability_violation",
                "severity": guardrail.severity,
                "summary": guardrail.rule,
            }
        )

    signals.sort(key=lambda item: SIGNAL_PRIORITY.get(str(item.get("signal") or ""), 99))
    return signals


def render_runtime_semantic_strip(*, thread_id: str) -> None:
    if str(os.environ.get("DADBOT_SHOW_RUNTIME_SIGNALS", "0")).strip().lower() not in {"1", "true", "yes", "on"}:
        return
    api = get_chat_event_api()
    resolved_thread_id = str(thread_id or api.active_thread_id or "default")
    view = load_thread_projection(api=api, thread_id=resolved_thread_id)
    guardrail = runtime_guardrail_view_from_session()
    signals = runtime_semantic_signal_snapshot(view=view, guardrail=guardrail)
    if not signals:
        return
    with st.container(border=True):
        st.markdown("**Runtime State Signals**")
        st.caption("Canonical Event to UI mapping for this thread.")
        for signal in signals:
            signal_name = str(signal.get("signal") or "runtime_state").replace("_", " ")
            source = str(signal.get("source") or "unknown")
            summary = str(signal.get("summary") or "").strip()
            if summary:
                st.caption(f"{signal_name.title()} from {source}: {summary}")
            else:
                st.caption(f"{signal_name.title()} from {source}")


def render_runtime_guardrails_card(*, dismiss_key: str = "dismiss-runtime-guardrails") -> None:
    guardrail = runtime_guardrail_view_from_session()
    if guardrail is None:
        return

    level_label = {
        "critical": "Critical",
        "high": "High",
        "medium": "Medium",
        "low": "Low",
    }.get(guardrail.severity, "Low")

    with st.container(border=True):
        st.markdown("**Runtime Guardrails**")
        st.caption(f"Severity: {level_label}")
        st.caption(f"Action: {guardrail.action}")
        st.caption(f"Rule: {guardrail.rule}")
        st.caption(f"Layer: {guardrail.layer}")
        st.caption(f"Why: {guardrail.reason}")
        st.caption(f"How to fix: {guardrail.fix_hint}")
        if guardrail.message:
            st.caption(f"Details: {guardrail.message}")
        if st.button("Dismiss Guardrail", key=dismiss_key, use_container_width=False):
            state_manager.clear_runtime_rejection()
            st.rerun()


def record_turn_inspector_from_runtime_result(
    *, thread_id: str, prompt: str, runtime_result: dict, view: ThreadView
) -> None:
    normalized_thread = str(thread_id or "default")
    prompt_text = str(prompt or "").strip()
    if prompt_text:
        record_turn_timeline_event(
            thread_id=normalized_thread,
            event_type="user_message",
            summary=prompt_text[:180],
            payload={"length": len(prompt_text)},
        )

    thinking = dict(view.thinking or {})
    if thinking:
        mood = str(thinking.get("mood_detected") or "neutral")
        final_path = str(thinking.get("final_path") or "model_reply")
        record_turn_timeline_event(
            thread_id=normalized_thread,
            event_type="thinking_update",
            summary=f"mood={mood}, path={final_path}",
            payload={"reply_source": str(thinking.get("reply_source") or "")},
        )

    decision = dict(view.decision or {})
    decisions = dict(decision.get("decisions") or {})
    decision_summary = ", ".join(key for key, value in decisions.items() if bool(value)) or "no effects selected"
    if decision or decisions:
        record_turn_timeline_event(
            thread_id=normalized_thread,
            event_type="decision_event",
            summary=decision_summary,
            payload={"decisions": decisions},
        )

    def _event_presence(value) -> bool:
        if isinstance(value, list):
            return len(value) > 0
        if isinstance(value, tuple):
            return len(value) > 0
        try:
            return int(value or 0) > 0
        except Exception:
            return bool(value)

    turn_graph = dict(view.turn_graph or {})
    events_by_type = dict(turn_graph.get("events_by_type") or {})
    if _event_presence(events_by_type.get("photo_request")):
        record_turn_timeline_event(
            thread_id=normalized_thread, event_type="photo_request", summary="photo effect requested"
        )
    if _event_presence(events_by_type.get("tts_request")):
        record_turn_timeline_event(
            thread_id=normalized_thread, event_type="tts_request", summary="tts effect requested"
        )

    reply = str(runtime_result.get("reply") or "").strip()
    if reply:
        record_turn_timeline_event(
            thread_id=normalized_thread,
            event_type="assistant_reply",
            summary=reply[:180],
            payload={"should_end": bool(runtime_result.get("should_end", False))},
        )


def render_agentic_trace(runtime_result: dict) -> None:
    """Render delegation/reasoning trace surfaced by runtime_result."""
    trace = dict(runtime_result.get("multi_agent_trace") or {})
    if not trace:
        return

    arbitration = dict(trace.get("arbitration") or {})
    blackboard = dict(trace.get("blackboard") or {})
    delegation_results = list(trace.get("delegation_results") or [])
    reasoning_steps = list(trace.get("reasoning_steps") or [])

    with st.expander("Delegation + Reasoning", expanded=False):
        subtasks = int(trace.get("subtasks_executed") or 0)
        depth = int(trace.get("delegation_depth") or 0)
        mode = str(arbitration.get("mode") or "sequential")
        st.caption(f"Mode: {mode} | Subtasks: {subtasks} | Depth: {depth}")

        if arbitration:
            st.caption("Supervisor arbitration")
            st.json(arbitration)

        if delegation_results:
            st.caption("Sub-agent outputs")
            for index, item in enumerate(delegation_results[:8], start=1):
                st.caption(f"{index}. {str(item or '').strip()[:220]}")

        if blackboard:
            st.caption("Shared blackboard")
            st.json(blackboard)

        if reasoning_steps:
            st.caption("Reasoning steps")
            for index, step in enumerate(reasoning_steps[:8], start=1):
                if isinstance(step, dict):
                    text = str(step.get("output") or step.get("result") or step.get("step") or "").strip()
                else:
                    text = str(step or "").strip()
                if text:
                    st.caption(f"{index}. {text[:220]}")


def render_turn_inspector(*, thread_id: str, view: ThreadView) -> None:
    normalized_thread = str(thread_id or "default")
    timeline = [item for item in runtime_turn_timeline() if str(item.get("thread_id") or "") == normalized_thread]
    thinking = dict(view.thinking or {})
    decision = dict(view.decision or {})
    if not timeline and not thinking and not decision:
        return

    with st.expander("Turn Inspector", expanded=False):
        st.caption("User message -> reasoning -> decision -> effects -> reply")

        rejection = st.session_state.get(RUNTIME_REJECTION_SESSION_KEY)
        if isinstance(rejection, dict):
            rejection_message = str(rejection.get("message") or "").strip()
            if "SaveNode commit boundary" in rejection_message or "outside SaveNode" in rejection_message:
                st.warning(f"Strict boundary guard tripped: {rejection_message}")

        if timeline:
            for item in timeline[-12:]:
                timestamp = str(item.get("timestamp") or "")
                event_type = str(item.get("event_type") or "event")
                summary = str(item.get("summary") or "")
                st.caption(f"{timestamp} | {event_type}: {summary}")

        rules = list(thinking.get("active_rules") or [])
        if rules:
            st.caption("Reasoning rules:")
            for rule in rules[:8]:
                st.caption(f"- {rule}")

        steps = list(thinking.get("pipeline_steps") or [])
        if steps:
            st.caption("Pipeline chain:")
            for step in steps[:10]:
                step_name = str(step.get("name") or "step")
                step_status = str(step.get("status") or "completed")
                step_detail = str(step.get("detail") or "")
                st.caption(f"- {step_name} [{step_status}] {step_detail}")

        observed_events = [
            str(item or "").strip().lower()
            for item in list(dict(view.turn_graph or {}).get("event_sequence") or [])
            if str(item or "").strip()
        ]
        persisted_stage_diagnostics = dict(dict(view.turn_graph or {}).get("stage_diagnostics") or {})
        if STAGE_METADATA:
            st.caption("Stage contract:")
            allowed_path = " -> ".join(STAGE_METADATA.keys())
            if allowed_path:
                st.caption(f"- Allowed path: {allowed_path}")
            if observed_events:
                st.caption(f"- Observed events: {' -> '.join(observed_events)}")
            trace_id = str(dict(view.turn_graph or {}).get("trace_id") or "").strip()
            if trace_id:
                st.caption(f"- Trace: {trace_id}")
            for stage_name, definition in STAGE_METADATA.items():
                stage_snapshot = dict(persisted_stage_diagnostics.get(stage_name) or {})
                if stage_snapshot:
                    guard_items = list(stage_snapshot.get("guards") or [])
                    guard_text = (
                        ", ".join(
                            f"{item.get('name')}={'ok' if item.get('passed') else 'blocked'}" for item in guard_items
                        )
                        or "none"
                    )
                    emit_text = ", ".join(stage_snapshot.get("allowed_emits") or []) or "none"
                    next_text = ", ".join(stage_snapshot.get("allowed_next") or []) or "end"
                    observed_text = ", ".join(stage_snapshot.get("observed_runtime_emits") or []) or "none observed"
                    purpose = str(stage_snapshot.get("purpose") or definition.purpose)
                else:
                    observed_for_stage = [event for event in observed_events if event in set(definition.emits)]
                    guard_text = ", ".join(definition.guards) if definition.guards else "none"
                    emit_text = ", ".join(definition.emits) if definition.emits else "none"
                    next_text = ", ".join(definition.next) if definition.next else "end"
                    observed_text = ", ".join(observed_for_stage) if observed_for_stage else "none observed"
                    purpose = definition.purpose
                st.caption(
                    f"- {stage_name}: {purpose} | guards={guard_text} | emits={emit_text} | next={next_text} | observed={observed_text}"
                )


def _extract_world_model_from_runtime(api: UIRuntimeAPI, *, thread_id: str) -> dict[str, Any] | None:
    orchestrator = getattr(api, "turn_orchestrator", None)
    turn_context = getattr(orchestrator, "_last_turn_context", None)
    if turn_context is None:
        return None

    metadata = dict(getattr(turn_context, "metadata", {}) or {})
    snapshot = dict(metadata.get("user_world_model") or {})
    if not snapshot:
        return None

    normalized_thread_id = str(thread_id or api.active_thread_id or "default")
    snapshot["thread_id"] = normalized_thread_id
    snapshot["captured_at"] = datetime.utcnow().isoformat() + "Z"
    return snapshot


def _record_world_model_snapshot(api: UIRuntimeAPI, *, thread_id: str) -> dict[str, Any] | None:
    snapshot = _extract_world_model_from_runtime(api, thread_id=thread_id)
    if not snapshot:
        return None

    st.session_state[WORLD_MODEL_SESSION_KEY] = dict(snapshot)
    history = list(st.session_state.get(WORLD_MODEL_HISTORY_SESSION_KEY) or [])
    history.append(dict(snapshot))
    st.session_state[WORLD_MODEL_HISTORY_SESSION_KEY] = history[-10:]
    return snapshot


def render_world_model_status_panel(api: UIRuntimeAPI) -> None:
    active_thread_id = str(api.active_thread_id or "default")
    _record_world_model_snapshot(api, thread_id=active_thread_id)

    with st.expander("User World Model (Current Turn)", expanded=False):
        snapshot = dict(st.session_state.get(WORLD_MODEL_SESSION_KEY) or {})
        if not snapshot:
            st.info("No world model snapshot yet - send a message first.")
            return

        col1, col2 = st.columns(2)
        trust_level = int(snapshot.get("trust_level", 0) or 0)
        openness_level = int(snapshot.get("openness_level", 0) or 0)

        def _band(level: int) -> tuple[str, str]:
            if level < 40:
                return "low", "red"
            if level <= 75:
                return "medium", "yellow"
            return "high", "green"

        with col1:
            st.metric("Trust Level", trust_level)
            st.metric("Openness", openness_level)
            st.metric("Momentum", str(snapshot.get("emotional_momentum", "")) or "unknown")
        with col2:
            st.metric("Policy Version", str(snapshot.get("policy_version", "unknown")) or "unknown")
            session_id = str(snapshot.get("session_id", ""))
            clipped_session = session_id[:8] + "..." if len(session_id) > 8 else (session_id or "-")
            st.metric("Session ID", clipped_session)

        trust_band, trust_color = _band(trust_level)
        openness_band, openness_color = _band(openness_level)
        st.caption(
            f"Trust band: {trust_band} ({trust_color}) | Openness band: {openness_band} ({openness_color})"
        )

        goals = [str(goal).strip() for goal in list(snapshot.get("active_goals") or []) if str(goal).strip()]
        contradictions = [
            str(item).strip() for item in list(snapshot.get("contradictions") or []) if str(item).strip()
        ]
        family_map = {
            str(key).strip(): str(value).strip()
            for key, value in dict(snapshot.get("family_map") or {}).items()
            if str(key).strip() and str(value).strip()
        }

        st.markdown("**Active Goals**")
        if goals:
            st.dataframe(
                [{"goal": goal} for goal in goals],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.caption("No extracted active goals yet.")

        st.markdown("**Contradictions**")
        if contradictions:
            for contradiction in contradictions[:8]:
                st.warning(contradiction)
        else:
            st.caption("No contradictions detected.")

        st.markdown("**Family Map**")
        if family_map:
            st.dataframe(
                [{"relation": relation, "descriptor": descriptor} for relation, descriptor in family_map.items()],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.caption("No family map entries yet.")

        st.json(snapshot, expanded=False)

    with st.expander("User World Model History (Last 10)", expanded=False):
        history = list(st.session_state.get(WORLD_MODEL_HISTORY_SESSION_KEY) or [])
        if not history:
            st.caption("No snapshots captured yet.")
            return

        trust_series = [int(dict(item).get("trust_level", 0) or 0) for item in history if isinstance(item, dict)]
        openness_series = [int(dict(item).get("openness_level", 0) or 0) for item in history if isinstance(item, dict)]
        if trust_series and openness_series:
            latest_trust = trust_series[-1]
            prior_trust = trust_series[-2] if len(trust_series) > 1 else latest_trust
            latest_openness = openness_series[-1]
            prior_openness = openness_series[-2] if len(openness_series) > 1 else latest_openness

            trend_col1, trend_col2 = st.columns(2)
            trend_col1.metric("Trust Trend", latest_trust, delta=latest_trust - prior_trust)
            trend_col2.metric("Openness Trend", latest_openness, delta=latest_openness - prior_openness)
            st.line_chart(
                {
                    "trust_level": trust_series[-10:],
                    "openness_level": openness_series[-10:],
                },
                use_container_width=True,
                height=160,
            )

        rows = [
            {
                "captured_at": str(item.get("captured_at") or ""),
                "thread_id": str(item.get("thread_id") or ""),
                "policy_version": str(item.get("policy_version") or ""),
                "trust_level": int(item.get("trust_level", 0) or 0),
                "openness_level": int(item.get("openness_level", 0) or 0),
                "emotional_momentum": str(item.get("emotional_momentum") or ""),
            }
            for item in history[-10:]
            if isinstance(item, dict)
        ]
        if not rows:
            st.caption("No valid snapshots in history.")
            return
        st.dataframe(rows[::-1], use_container_width=True, hide_index=True)


def ui_shell_snapshot(bot):
    # Call DadBot's own method, not the event API
        # Always call the method on the DadBot instance, never via the event API
        print(f"[DEBUG] ui_shell_snapshot called with bot type: {type(bot)}", file=sys.stderr)
        print(f"[DEBUG] ui_shell_snapshot source:\n{inspect.getsource(ui_shell_snapshot)}", file=sys.stderr)
        return bot.ui_shell_snapshot() if bot is not None else {}


def voice_profile_payload(voice: dict) -> dict:
    keys = {
        "enabled",
        "mode",
        "auto_send_always_listening",
        "wake_word_required",
        "wake_word_phrase",
        "stt_enabled",
        "stt_backend",
        "stt_model",
        "stt_language",
        "tts_enabled",
        "tts_autoplay",
        "tts_voice",
        "tts_backend",
        "tts_piper_model_path",
        "tts_rate",
        "warmth",
        "dad_joke_frequency",
        "pacing",
        "muted",
        "mic_preference",
        "last_used_device",
        "auto_listen_allowed",
        "last_mode",
        "interruptions_enabled",
        "barge_in_enabled",
        "barge_in_min_audio_bytes",
        "allow_tts_cancel",
        "priority_override_enabled",
        "known_device_ids",
    }
    return {key: voice.get(key) for key in sorted(keys)}


def persist_voice_profile_if_changed(bot, voice: dict) -> None:
    api = get_chat_event_api()
    payload = voice_profile_payload(voice)
    current = api.PROFILE.get("voice", {}) if isinstance(api.PROFILE, dict) else {}
    if not isinstance(current, dict):
        current = {}
    if all(current.get(key) == value for key, value in payload.items()):
        return
    api.update_voice_profile(payload)
    api.save_profile()


def get_voice_session_controller(bot) -> VoiceSessionController:
    api = get_chat_event_api()
    runtime_state = st.session_state.setdefault("voice_runtime_state", {})
    controller = VoiceSessionController(
        voice_preferences(),
        runtime_state=runtime_state,
        ledger_emitter=lambda event_type, payload: emit_voice_runtime_ledger_event(event_type, payload),
        session_id_provider=lambda: str(api.active_thread_id or "voice-ui"),
        trace_id_provider=lambda: st.session_state.setdefault("voice_trace_id", uuid.uuid4().hex),
    )
    if "voice_profile_fingerprint" not in st.session_state:
        st.session_state.voice_profile_fingerprint = hashlib.sha1(
            json.dumps(voice_profile_payload(controller.voice_config), sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
    return controller


@st.cache_resource(show_spinner=False)
def get_chat_event_api() -> UIRuntimeAPI:
    return interaction_controller.get_chat_event_api()


@st.cache_resource(show_spinner=False)
def get_runtime() -> StreamlitRuntime:
    return interaction_controller.get_runtime()


def process_prompt_via_runtime(
    *,
    thread_id: str,
    prompt: str,
    attachments: list[dict] | None = None,
    model_override: str = "",
    temperature_style: str = "balanced",
    message_metadata: dict | None = None,
    turn_controls: dict | None = None,
    adaptive_compute: bool = True,
) -> dict:
    started = time.perf_counter()
    result = interaction_controller.process_prompt_via_runtime(
        thread_id=thread_id,
        prompt=prompt,
        attachments=attachments,
        model_override=model_override,
        temperature_style=temperature_style,
        message_metadata=message_metadata,
        turn_controls=turn_controls,
        adaptive_compute=adaptive_compute,
    )
    _ = int(max(0.0, (time.perf_counter() - started) * 1000.0))
    return result


def render_progressive_reply(reply: str, *, enable_streaming: bool = True) -> None:
    text = str(reply or "")
    if not text:
        st.markdown("")
        return
    if not enable_streaming or not hasattr(st, "write_stream"):
        st.markdown(text)
        return

    # Word-level chunking keeps UI responsive while preserving deterministic final text.
    tokens = text.split(" ")

    def _gen():
        for idx, token in enumerate(tokens):
            suffix = " " if idx < len(tokens) - 1 else ""
            yield token + suffix

    st.write_stream(_gen())


def emit_voice_runtime_ledger_event(event_type: str, payload: dict) -> None:
    interaction_controller.emit_voice_runtime_ledger_event(event_type, payload)


def purge_session_context(bot, mode="full"):
    return interaction_controller.purge_session_context(bot=bot, initialize_session=initialize_session, mode=mode)


def summarize_document_attachment(uploaded_file, max_chars=420):
    if uploaded_file is None:
        return None
    raw_bytes = b""
    try:
        raw_bytes = uploaded_file.getvalue() or b""
    except Exception:
        raw_bytes = b""
    if not raw_bytes:
        return None

    name = str(getattr(uploaded_file, "name", "document")).strip() or "document"
    mime_type = str(getattr(uploaded_file, "type", "")).strip() or (mimetypes.guess_type(name)[0] or "")
    text_excerpt = ""
    if mime_type.startswith("text/") or name.lower().endswith((".txt", ".md", ".csv", ".json", ".py", ".log")):
        text_excerpt = raw_bytes.decode("utf-8", errors="replace").strip()
    elif name.lower().endswith(".pdf"):
        text_excerpt = f"PDF uploaded ({len(raw_bytes)} bytes)."
    else:
        text_excerpt = f"File uploaded ({len(raw_bytes)} bytes)."

    cleaned = " ".join(text_excerpt.split())
    if len(cleaned) > max_chars:
        cleaned = cleaned[: max_chars - 3].rstrip() + "..."
    if not cleaned:
        cleaned = f"File uploaded ({len(raw_bytes)} bytes)."

    return {
        "type": "document",
        "name": name,
        "mime_type": mime_type,
        "note": f"Tony shared a document: {name}",
        "text": cleaned,
    }


def build_chat_attachments_from_uploads(uploaded_files):
    attachments = []
    issues = []
    for uploaded_file in list(uploaded_files or []):
        if uploaded_file is None:
            continue
        file_name = str(getattr(uploaded_file, "name", "upload")).strip().lower()
        mime_type = str(getattr(uploaded_file, "type", "")).strip().lower()
        raw_bytes = b""
        try:
            raw_bytes = uploaded_file.getvalue() or b""
        except Exception as exc:
            raw_bytes = b""
            issues.append(f"{getattr(uploaded_file, 'name', 'upload')}: could not read file ({exc}).")
        if not raw_bytes:
            issues.append(f"{getattr(uploaded_file, 'name', 'upload')}: file was empty.")
            continue
        if len(raw_bytes) > MAX_UPLOAD_BYTES:
            issues.append(
                f"{getattr(uploaded_file, 'name', 'upload')}: file is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)}MB limit."
            )
            continue

        is_image = mime_type.startswith("image/") or file_name.endswith((".png", ".jpg", ".jpeg", ".webp"))
        if is_image:
            attachments.append(
                {
                    "type": "image",
                    "name": str(getattr(uploaded_file, "name", "image")),
                    "mime_type": mime_type,
                    "image_b64": base64.b64encode(raw_bytes).decode("utf-8"),
                    "note": f"Tony uploaded {getattr(uploaded_file, 'name', 'an image')!s}",
                }
            )
            continue

        summarized = summarize_document_attachment(uploaded_file)
        if summarized is not None:
            attachments.append(summarized)
        else:
            issues.append(f"{getattr(uploaded_file, 'name', 'upload')}: unsupported or unreadable document.")

    return attachments[:6], issues


def synthesize_piper_audio(text: str, *, model_path: str, rate: float = 1.0) -> tuple[bytes | None, str]:
    """Synthesize speech via the Piper TTS subprocess. Returns (audio_bytes, error)."""
    piper_exe = shutil.which("piper")
    if not piper_exe:
        return None, "Piper executable not found. Install from https://github.com/rhasspy/piper"
    model_path = str(model_path or "").strip()
    if not model_path or not Path(model_path).exists():
        return None, (
            "Piper model file not configured or not found. "
            "Download a .onnx model from https://rhasspy.github.io/piper-samples/ "
            "and set the path in Preferences → Voice → Piper model path."
        )
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            temp_path = handle.name
        result = subprocess.run(
            [piper_exe, "--model", model_path, "--output_file", temp_path],
            input=text.encode("utf-8"),
            capture_output=True,
            timeout=45,
        )
        if result.returncode != 0:
            stderr = result.stderr.decode("utf-8", errors="replace")[:200]
            return None, f"Piper failed (exit {result.returncode}): {stderr}"
        audio_bytes = Path(temp_path).read_bytes()
        return audio_bytes, ""
    except subprocess.TimeoutExpired:
        return None, "Piper timed out after 45 seconds."
    except Exception as exc:
        return None, f"Piper error: {exc}"
    finally:
        if temp_path:
            try:
                Path(temp_path).unlink(missing_ok=True)
            except Exception:
                pass


@st.cache_resource(show_spinner=False)
def load_faster_whisper_model(model_name):
    if WhisperModel is None:
        raise RuntimeError("faster-whisper is unavailable")
    return WhisperModel(model_name, device="cpu", compute_type="int8")


@st.cache_resource(show_spinner=False)
def load_openai_whisper_model(model_name):
    if openai_whisper is None:
        raise RuntimeError("openai-whisper is unavailable")
    return openai_whisper.load_model(model_name)


def transcribe_audio_bytes(audio_bytes, *, backend, model_name="base", language="en"):
    if not audio_bytes:
        return "", "No audio data received."

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            handle.write(audio_bytes)
            temp_path = handle.name

        if backend == "faster_whisper":
            model = load_faster_whisper_model(model_name)
            segments, _info = model.transcribe(temp_path, language=language or None, vad_filter=True, beam_size=1)
            transcript = " ".join((segment.text or "").strip() for segment in segments).strip()
            return transcript, ""

        if backend == "openai_whisper":
            model = load_openai_whisper_model(model_name)
            result = model.transcribe(temp_path, language=language or None, fp16=False)
            transcript = str((result or {}).get("text") or "").strip()
            return transcript, ""

        return "", f"Unsupported STT backend: {backend}"
    except Exception as exc:
        return "", f"Transcription failed: {exc}"
    finally:
        if temp_path:
            try:
                Path(temp_path).unlink(missing_ok=True)
            except Exception:
                pass


def _voice_known_devices(voice: dict, runtime_state: dict) -> list[str]:
    known = list(voice.get("known_device_ids") or [])
    runtime_known = list(runtime_state.get("known_devices") or [])
    merged = ["default"]
    for item in [*known, *runtime_known, str(voice.get("last_used_device") or "default")]:
        normalized = str(item or "").strip()
        if not normalized:
            continue
        if normalized not in merged:
            merged.append(normalized)
    return merged[:12]


def _persist_known_devices(voice: dict, known_devices: list[str]) -> None:
    cleaned = []
    for item in list(known_devices or []):
        normalized = str(item or "").strip()
        if normalized and normalized not in cleaned:
            cleaned.append(normalized)
    if "default" not in cleaned:
        cleaned.insert(0, "default")
    voice["known_device_ids"] = cleaned[:12]


def _collect_webrtc_audio_bytes(webrtc_ctx, *, key: str, min_bytes: int) -> bytes:
    if not webrtc_ctx or not getattr(getattr(webrtc_ctx, "state", None), "playing", False):
        return b""
    receiver = getattr(webrtc_ctx, "audio_receiver", None)
    if receiver is None:
        return b""

    try:
        frames = receiver.get_frames(timeout=0)
    except Exception:
        frames = []

    if not frames:
        return b""

    chunks = []
    for frame in frames:
        try:
            chunks.append(frame.to_ndarray().tobytes())
        except Exception:
            continue
    merged = b"".join(chunks)
    if not merged:
        return b""

    buffer_key = f"{key}:buffer"
    buffer_bytes = st.session_state.get(buffer_key, b"")
    if not isinstance(buffer_bytes, (bytes, bytearray)):
        buffer_bytes = b""
    buffer_bytes = bytes(buffer_bytes) + merged
    if len(buffer_bytes) < max(1, int(min_bytes or 1)):
        st.session_state[buffer_key] = buffer_bytes
        return b""

    st.session_state[buffer_key] = b""
    return buffer_bytes


def _render_voice_capture_layer(controller: VoiceSessionController, voice: dict, *, key_prefix: str) -> bytes:
    min_audio_bytes = max(3000, int(voice.get("barge_in_min_audio_bytes") or 3500))
    runtime_state = controller.runtime_state if isinstance(controller.runtime_state, dict) else {}

    known_devices = _voice_known_devices(voice, runtime_state)
    selected_device = st.selectbox(
        "Input device ID",
        options=known_devices,
        index=known_devices.index(str(voice.get("last_used_device") or "default"))
        if str(voice.get("last_used_device") or "default") in known_devices
        else 0,
        key=f"{key_prefix}-device-select",
        help="Persistent WebRTC device ID. Use default unless you need a specific microphone.",
    )

    custom_device = st.text_input(
        "Add device ID",
        value="",
        key=f"{key_prefix}-device-custom",
        placeholder="Paste a browser deviceId token (optional)",
    )
    if st.button("Save device ID", key=f"{key_prefix}-device-save", use_container_width=True):
        custom = str(custom_device or "").strip()
        if custom:
            known_devices.append(custom)
            _persist_known_devices(voice, known_devices)
            controller.set_device(custom)
            st.rerun()

    _persist_known_devices(voice, known_devices)
    voice["mic_preference"] = str(selected_device or "default")
    controller.set_device(selected_device)

    if not _WEBRTC_AVAILABLE:
        audio_label = (
            "Hold mic, speak, release"
            if str(voice.get("mode") or "push_to_talk") == "push_to_talk"
            else "Always-listening capture"
        )
        st.caption("WebRTC unavailable; using Streamlit audio input fallback.")
        clip = st.audio_input(audio_label, key=f"{key_prefix}-audio-fallback")
        if clip is None:
            return b""
        return bytes(clip.getvalue() or b"")

    media_audio: dict | bool = True
    if str(selected_device or "default") != "default":
        media_audio = {"deviceId": {"exact": str(selected_device)}}

    rtc_config = WebRtcRTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
    try:
        webrtc_ctx = webrtc_streamer(
            key=f"{key_prefix}-webrtc-capture",
            mode=WebRtcMode.SENDONLY,
            rtc_configuration=rtc_config,
            media_stream_constraints={"video": False, "audio": media_audio},
            async_processing=True,
        )
    except Exception:
        st.caption("WebRTC runtime unavailable in this environment; using audio input fallback.")
        clip = st.audio_input("Hold mic, speak, release", key=f"{key_prefix}-audio-fallback")
        if clip is None:
            return b""
        return bytes(clip.getvalue() or b"")

    if not (webrtc_ctx and webrtc_ctx.state.playing):
        st.caption("Click START to begin WebRTC microphone capture.")
        return b""

    return _collect_webrtc_audio_bytes(webrtc_ctx, key=f"{key_prefix}-webrtc-capture", min_bytes=min_audio_bytes)


def resolve_tts_voice_id(engine, voice_profile):
    profile = str(voice_profile or "warm_dad").strip().lower()
    profile_hints = {
        "warm_dad": ("david", "mark", "male", "guy", "father", "adult"),
        "deep_dad": ("david", "mark", "male", "deep", "baritone", "adult"),
        "gentle_dad": ("gentle", "calm", "soothing", "male", "david", "adult"),
    }
    hints = profile_hints.get(profile, profile_hints["warm_dad"])
    voices = list(engine.getProperty("voices") or [])

    scored = []
    for voice in voices:
        name = str(getattr(voice, "name", "") or "").lower()
        voice_id = str(getattr(voice, "id", "") or "").lower()
        blob = f"{name} {voice_id}"
        score = sum(1 for hint in hints if hint in blob)
        scored.append((score, getattr(voice, "id", "")))

    scored.sort(key=lambda item: item[0], reverse=True)
    best_id = scored[0][1] if scored and scored[0][1] else ""
    return best_id


def synthesize_tts_audio(reply_text, *, voice_profile="warm_dad", rate_delta=0, pacing=50):
    if pyttsx3 is None:
        return None, "pyttsx3 is not installed."
    text = str(reply_text or "").strip()
    if not text:
        return None, ""

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            temp_path = handle.name

        engine = pyttsx3.init()
        voice_id = resolve_tts_voice_id(engine, voice_profile)
        if voice_id:
            engine.setProperty("voice", voice_id)

        default_rate = int(cast(Any, engine.getProperty("rate")) or 180)
        pace_delta = int((max(0, min(100, int(pacing or 50))) - 50) * 0.6)
        engine.setProperty("rate", max(120, min(240, default_rate + int(rate_delta or 0) + pace_delta)))
        engine.save_to_file(text, temp_path)
        engine.runAndWait()

        audio_bytes = Path(temp_path).read_bytes()
        return audio_bytes, ""
    except Exception as exc:
        return None, f"TTS failed: {exc}"
    finally:
        if temp_path:
            try:
                Path(temp_path).unlink(missing_ok=True)
            except Exception:
                pass


@maybe_fragment
def render_voice_controls(bot: "DadBot"):
    controller = get_voice_session_controller(bot)
    voice = controller.voice_config
    snapshot = controller.snapshot()

    with st.container(border=True):
        st.subheader("Voice Control Plane")
        st.caption("One voice state machine controls listening, recording, processing, and speaking.")

        mode = st.radio(
            "Voice mode",
            options=["push_to_talk", "always_listening", "ambient"],
            index={"push_to_talk": 0, "always_listening": 1, "ambient": 2}.get(
                str(snapshot.get("mode") or "push_to_talk"), 0
            ),
            format_func=lambda value: {
                "push_to_talk": "Push-to-talk",
                "always_listening": "Always listening",
                "ambient": "Ambient (advanced)",
            }[value],
            horizontal=True,
            key="voice-mode-radio",
        )
        if mode != str(snapshot.get("mode") or "push_to_talk"):
            controller.set_mode(mode)
            persist_voice_profile_if_changed(bot, voice)
            snapshot = controller.snapshot()

        status_col1, status_col2 = st.columns(2)
        status_col1.markdown(f"**Status:** {snapshot.get('status', 'Idle')}")
        status_col2.markdown(f"**Mic:** {'Ready' if snapshot.get('mic_available') else 'Waiting'}")

        interruption_col1, interruption_col2 = st.columns(2)
        interruption_col1.checkbox(
            "Interruptions enabled",
            value=bool(voice.get("interruptions_enabled", True)),
            key="voice-interruptions-enabled",
        )
        interruption_col2.checkbox(
            "Barge-in enabled",
            value=bool(voice.get("barge_in_enabled", True)),
            key="voice-barge-in-enabled",
        )
        voice["interruptions_enabled"] = bool(st.session_state.get("voice-interruptions-enabled", True))
        voice["barge_in_enabled"] = bool(st.session_state.get("voice-barge-in-enabled", True))
        voice["allow_tts_cancel"] = st.checkbox(
            "Allow TTS cancel",
            value=bool(voice.get("allow_tts_cancel", True)),
            key="voice-allow-tts-cancel",
        )
        voice["barge_in_min_audio_bytes"] = int(
            st.slider(
                "Barge-in sensitivity",
                min_value=1000,
                max_value=12000,
                value=int(voice.get("barge_in_min_audio_bytes", 4000) or 4000),
                step=500,
                key="voice-barge-min-bytes",
                help="Lower values interrupt faster; higher values require stronger input before canceling current speech.",
            )
        )

        with st.expander("Voice debug", expanded=False):
            st.write(
                {
                    "mode": snapshot.get("mode"),
                    "state": snapshot.get("state"),
                    "last_event": snapshot.get("last_event"),
                    "last_error": snapshot.get("last_error"),
                    "safety_flags": snapshot.get("safety_flags"),
                    "last_used_device": snapshot.get("last_used_device"),
                    "known_devices": snapshot.get("known_devices"),
                    "transition_version": snapshot.get("transition_version"),
                    "cancel_requested": snapshot.get("cancel_requested"),
                    "cancel_reason": snapshot.get("cancel_reason"),
                    "priority_override": snapshot.get("priority_override"),
                }
            )

        action_col1, action_col2 = st.columns(2)
        if action_col1.button(
            "Enable voice" if not bool(voice.get("enabled")) else "Disable voice",
            key="voice-enabled-toggle",
            use_container_width=True,
        ):
            controller.set_enabled(not bool(voice.get("enabled")))
            persist_voice_profile_if_changed(bot, voice)
            st.rerun()

        if action_col2.button(
            "Unmute" if bool(snapshot.get("muted")) else "Mute",
            key="voice-mute-toggle",
            use_container_width=True,
        ):
            controller.set_muted(not bool(snapshot.get("muted")))
            persist_voice_profile_if_changed(bot, voice)
            snapshot = controller.snapshot()

        if not bool(voice.get("enabled")):
            st.info("Voice is disabled. Enable it to start recording.")
            return None

        interrupt_col1, interrupt_col2 = st.columns(2)
        if interrupt_col1.button("Cancel active voice turn", key="voice-cancel-active", use_container_width=True):
            controller.request_cancel(reason="manual_cancel", priority="high")
            snapshot = controller.snapshot()
        if interrupt_col2.button("Priority override: high", key="voice-priority-high", use_container_width=True):
            controller.apply_priority_override("high", reason="ui_override")
            snapshot = controller.snapshot()

        stt_ready, stt_backend, stt_message = local_stt_backend_status(voice)
        if not stt_ready:
            st.warning(stt_message)
            render_voice_dependency_help(context_key="chat-stt")
            controller.mark_error(stt_message)
            return None
        controller.mark_mic_available(True)

        tts_ready, _tts_backend, tts_message = local_tts_backend_status()
        if bool(voice.get("tts_enabled", True)) and not tts_ready:
            st.warning(tts_message)
            render_voice_dependency_help(context_key="chat-tts")

        if str(mode) == "ambient":
            st.caption("Ambient mode active: capture runs in the background listener loop.")
            persist_voice_profile_if_changed(bot, voice)
            return None

        audio_bytes = _render_voice_capture_layer(controller, voice, key_prefix="voice-control")
        if not audio_bytes:
            persist_voice_profile_if_changed(bot, voice)
            return None
        controller.start_turn(turn_id=uuid.uuid4().hex, priority=str(snapshot.get("priority_override") or "normal"))
        clip_hash = hashlib.sha1(audio_bytes).hexdigest() if audio_bytes else ""
        transcript_key = f"voice-transcript:{clip_hash}"
        transcript_text = str(st.session_state.get(transcript_key) or "")
        transcript_error = ""

        if not transcript_text:
            with st.spinner("Transcribing locally..."):
                transcript_text, transcript_error = controller.process_audio_capture(
                    audio_bytes,
                    transcribe_fn=lambda payload: transcribe_audio_bytes(
                        payload,
                        backend=stt_backend,
                        model_name=str(voice.get("stt_model") or "base"),
                        language=str(voice.get("stt_language") or "en"),
                    ),
                )
            if transcript_text:
                st.session_state[transcript_key] = transcript_text

        if transcript_error:
            st.error(transcript_error)
            persist_voice_profile_if_changed(bot, voice)
            return None
        if not transcript_text:
            persist_voice_profile_if_changed(bot, voice)
            return None

        edited = st.text_area(
            "Transcript",
            value=transcript_text,
            key=f"voice-transcript-edit:{clip_hash}",
            height=80,
        )
        edited = str(edited or "").strip()
        if not edited:
            persist_voice_profile_if_changed(bot, voice)
            return None

        if mode == "always_listening" and bool(voice.get("auto_send_always_listening", True)):
            st.info("Always-listening captured and queued this utterance.")
            persist_voice_profile_if_changed(bot, voice)
            return edited

        if st.button("Send transcript", key=f"voice-send:{clip_hash}", type="primary"):
            persist_voice_profile_if_changed(bot, voice)
            return edited
        persist_voice_profile_if_changed(bot, voice)
        return None


# ====================== REAL-TIME WEBRTC VOICE CALL ======================
# ====================== AMBIENT VOICE LISTENER ===========================


@ambient_fragment(run_every=2)
def render_ambient_voice_listener(bot: "DadBot"):
    """Hands-free continuous listener fragment — reruns every 2 s automatically.

    Uses Streamlit's ``st.fragment(run_every=N)`` to auto-refresh the audio
    capture widget and drain a thread-safe utterance queue without requiring
    any user interaction.
    """
    controller = get_voice_session_controller(bot)
    voice = controller.voice_config
    if not bool(voice.get("enabled")) or str(voice.get("mode") or "") != "ambient":
        return

    st.markdown(
        "<div style='display:flex;align-items:center;gap:0.6rem;'>"
        "<span style='width:10px;height:10px;border-radius:999px;background:#22c55e;"
        "animation:pulse 1.5s infinite;display:inline-block;'></span>"
        "<span style='font-size:0.85rem;opacity:0.8;'>Ambient listener active — speak naturally</span>"
        "</div>",
        unsafe_allow_html=True,
    )

    stt_ready, stt_backend, stt_message = local_stt_backend_status(voice)
    if not stt_ready:
        controller.mark_error(stt_message)
        st.caption(f"STT offline: {stt_message}")
        return
    controller.mark_mic_available(True)

    audio_bytes = _render_voice_capture_layer(controller, voice, key_prefix="ambient-voice")
    if not audio_bytes or len(audio_bytes) < 4000:
        return

    controller.start_turn(turn_id=uuid.uuid4().hex, priority="normal")

    clip_hash = hashlib.sha1(audio_bytes).hexdigest()
    processed_key = f"ambient-processed:{clip_hash}"
    if st.session_state.get(processed_key):
        return  # already handled this exact clip

    st.session_state[processed_key] = True

    transcript_text, transcript_error = controller.process_audio_capture(
        audio_bytes,
        transcribe_fn=lambda payload: transcribe_audio_bytes(
            payload,
            backend=stt_backend,
            model_name=str(voice.get("stt_model") or "base"),
            language=str(voice.get("stt_language") or "en"),
        ),
    )
    if transcript_error or not transcript_text:
        persist_voice_profile_if_changed(bot, voice)
        return

    # Queue the utterance — main chat tab drains this on its next render
    queue = st.session_state.setdefault("ambient_utterance_queue", [])
    queue.append(transcript_text.strip())
    st.toast(f'Dad heard: "{transcript_text[:60]}"', icon="🎙️")
    persist_voice_profile_if_changed(bot, voice)


_WEBRTC_EXPERIMENTAL_ENABLED = str(os.environ.get("DADBOT_ENABLE_EXPERIMENTAL_WEBRTC_CALL") or "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


@maybe_fragment
def render_realtime_voice_call(bot: "DadBot"):
    """Live, hands-free two-way voice call powered by WebRTC + faster-whisper STT + Piper/pyttsx3 TTS."""
    controller = get_voice_session_controller(bot)
    voice = controller.voice_config

    if not _WEBRTC_EXPERIMENTAL_ENABLED:
        st.info(
            "Real-time WebRTC calling is currently disabled for stability. "
            "Use the Voice input panel in Chat (push-to-talk) for reliable local STT/TTS. "
            "Set DADBOT_ENABLE_EXPERIMENTAL_WEBRTC_CALL=1 to re-enable this experimental mode."
        )
        return

    if not _WEBRTC_AVAILABLE:
        st.info("Install `streamlit-webrtc` to enable real-time voice calls:\n```\npip install streamlit-webrtc\n```")
        return

    if not voice.get("enabled", False):
        st.info("Enable voice in Preferences → Voice to use the real-time call feature.")
        return

    st.subheader("📞 Talk to Dad — Live")
    st.caption("Hands-free, real-time voice conversation. Uses your mic → STT → Dad → TTS pipeline.")

    known_devices = _voice_known_devices(
        voice, controller.runtime_state if isinstance(controller.runtime_state, dict) else {}
    )
    selected_device = st.selectbox(
        "Realtime call input device ID",
        options=known_devices,
        index=known_devices.index(str(voice.get("last_used_device") or "default"))
        if str(voice.get("last_used_device") or "default") in known_devices
        else 0,
        key="webrtc-call-device",
    )
    controller.set_device(selected_device)
    _persist_known_devices(voice, known_devices)

    rtc_config = WebRtcRTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

    media_audio: dict | bool = True
    if str(selected_device or "default") != "default":
        media_audio = {"deviceId": {"exact": str(selected_device)}}

    try:
        webrtc_ctx = webrtc_streamer(
            key="dadbot-voice-call",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=rtc_config,
            media_stream_constraints={"video": False, "audio": media_audio},
            async_processing=True,
        )
    except Exception:
        st.caption("WebRTC runtime unavailable in this environment; using audio input fallback.")
        return

    if not (webrtc_ctx and webrtc_ctx.state.playing):
        st.caption("Click **START** above, allow microphone access, then speak naturally.")
        return

    st.success("Live call connected — speak naturally!")

    # ── Capture + transcribe audio frames ────────────────────────────────────
    audio_receiver = getattr(webrtc_ctx, "audio_receiver", None)
    if audio_receiver:
        try:
            frames = audio_receiver.get_frames(timeout=0)
        except Exception:
            frames = []
        if frames:
            chunks = []
            for frame in frames:
                try:
                    arr = frame.to_ndarray()
                    chunks.append(arr.tobytes())
                except Exception:
                    pass
            audio_bytes = b"".join(chunks)
            # Only transcribe when we have a meaningful audio buffer (~0.5 s at 16 kHz mono)
            if len(audio_bytes) > 8000:
                stt_ready, stt_backend, _ = local_stt_backend_status(voice)
                if stt_ready:
                    transcript, _err = controller.process_audio_capture(
                        audio_bytes,
                        transcribe_fn=lambda payload: transcribe_audio_bytes(
                            payload,
                            backend=stt_backend,
                            model_name=str(voice.get("stt_model") or "base"),
                            language=str(voice.get("stt_language") or "en"),
                        ),
                        min_audio_bytes=8000,
                    )
                    if transcript and transcript.strip():
                        st.session_state["realtime_transcript"] = transcript.strip()

    # ── Show live transcript ──────────────────────────────────────────────────
    live_transcript = str(st.session_state.get("realtime_transcript") or "").strip()
    if live_transcript:
        st.markdown(f"**You said:** {live_transcript}")

    # ── Send to Dad ───────────────────────────────────────────────────────────
    send_col, clear_col = st.columns([3, 1])
    if send_col.button("Send to Dad", type="primary", key="webrtc-send-btn", use_container_width=True):
        prompt = str(st.session_state.get("realtime_transcript") or "").strip()
        if prompt:
            with st.spinner("Dad is thinking..."):
                try:
                    runtime_result = process_prompt_via_runtime(
                        thread_id=str(get_chat_event_api().active_thread_id or "default"),
                        prompt=prompt,
                        attachments=[],
                    )
                    reply_text = str(runtime_result.get("reply") or "")

                    if reply_text:
                        st.markdown(f"**Dad:** {reply_text}")
                        render_reply_tts(bot, reply_text)
                except Exception as exc:
                    record_runtime_rejection(exc, action="realtime_voice_send")
                    st.error(f"Dad Runtime blocked or failed this request: {exc}")

            st.session_state["realtime_transcript"] = ""
        else:
            st.warning("Nothing transcribed yet — speak first, then click Send.")
    if clear_col.button("Clear", key="webrtc-clear-btn", use_container_width=True):
        st.session_state["realtime_transcript"] = ""
        st.rerun()

    st.caption("Tip: speak naturally, wait a moment for transcription, then press Send.")


def render_reply_tts(bot: "DadBot", reply_text):
    controller = get_voice_session_controller(bot)
    voice = controller.voice_config
    if not bool(voice.get("enabled")) or not bool(voice.get("tts_enabled", True)) or bool(voice.get("muted", False)):
        return

    cancel_state = controller.consume_cancel()
    if bool(cancel_state.get("cancel_requested")) and bool(voice.get("allow_tts_cancel", True)):
        controller.complete_turn()
        return

    text = str(reply_text or "").strip()
    if not text:
        return

    cache = st.session_state.setdefault("voice_tts_cache", {})
    key = hashlib.sha1(
        (text + "|" + str(voice.get("tts_voice") or "warm_dad") + "|" + str(int(voice.get("tts_rate") or 0))).encode(
            "utf-8"
        )
    ).hexdigest()

    audio_bytes = cache.get(key)
    error = ""
    if audio_bytes is None:
        tts_backend = str(voice.get("tts_backend") or "pyttsx3").strip().lower()
        piper_model = str(voice.get("tts_piper_model_path") or "").strip()
        controller.begin_speaking()
        with st.spinner("Generating local Dad voice audio..."):
            if tts_backend == "piper" or (tts_backend == "auto" and shutil.which("piper") and piper_model):
                audio_bytes, error = synthesize_piper_audio(text, model_path=piper_model)
            else:
                audio_bytes, error = synthesize_tts_audio(
                    text,
                    voice_profile=str(voice.get("tts_voice") or "warm_dad"),
                    rate_delta=int(voice.get("tts_rate") or 0),
                    pacing=int(voice.get("pacing", 50) or 50),
                )
        if audio_bytes:
            cache[key] = audio_bytes
        if error:
            controller.mark_error(error)
        else:
            controller.complete_turn()

    if error:
        st.warning(error)
        persist_voice_profile_if_changed(bot, voice)
        return
    if audio_bytes:
        pending_cancel = controller.consume_cancel()
        if bool(pending_cancel.get("cancel_requested")) and bool(voice.get("allow_tts_cancel", True)):
            controller.complete_turn()
            persist_voice_profile_if_changed(bot, voice)
            return
        st.audio(audio_bytes, format="audio/wav", autoplay=bool(voice.get("tts_autoplay", False)))
        controller.complete_turn()
    persist_voice_profile_if_changed(bot, voice)


def theme_palette(preferences, mood="neutral"):
    mode = str(preferences.get("theme_mode") or "warm").strip().lower()
    normalized_mood = str(mood or "neutral").strip().lower()
    # mood_override: when auto_mood_theme is on, pick a fully distinct palette
    mood_override = None
    if bool(preferences.get("auto_mood_theme", True)):
        if normalized_mood == "sad":
            mood_override = "sad"
        elif normalized_mood == "stressed":
            mood_override = "stressed"
        elif normalized_mood == "frustrated":
            mood_override = "frustrated"
        elif normalized_mood in {"tired"}:
            mode = "night"
        elif normalized_mood in {"positive"} and mode != "night":
            mode = "warm"
    high_contrast = bool(preferences.get("high_contrast"))
    palettes = {
        "warm": {
            "primary": "#d45d1f",
            "accent": "#0f766e",
            "bg": "#f5f2eb",
            "surface": "#fffdf8",
            "surface_alt": "#f2e8d9",
            "text": "#1d2730",
            "muted": "#5e6772",
            "hero_start": "#0f2c3a",
            "hero_end": "#8a3f1d",
            "border": "rgba(29, 39, 48, 0.14)",
        },
        "night": {
            "primary": "#fb923c",
            "accent": "#22d3ee",
            "bg": "#0e1622",
            "surface": "#152334",
            "surface_alt": "#1d3147",
            "text": "#e7eef7",
            "muted": "#acbbc9",
            "hero_start": "#10283c",
            "hero_end": "#3c2a54",
            "border": "rgba(231, 238, 247, 0.14)",
        },
        # Mood-specific palettes
        "sad": {
            "primary": "#3b82f6",
            "accent": "#0284c7",
            "bg": "#edf4ff",
            "surface": "#f7fbff",
            "surface_alt": "#deebff",
            "text": "#17263a",
            "muted": "#5e7690",
            "hero_start": "#21364e",
            "hero_end": "#395f8e",
            "border": "rgba(23, 38, 58, 0.14)",
        },
        "stressed": {
            "primary": "#dc2626",
            "accent": "#ea580c",
            "bg": "#fff3f0",
            "surface": "#fff9f8",
            "surface_alt": "#ffe4db",
            "text": "#311f1d",
            "muted": "#7a5b56",
            "hero_start": "#3e1d1b",
            "hero_end": "#8b2d22",
            "border": "rgba(49, 31, 29, 0.14)",
        },
        "frustrated": {
            "primary": "#be123c",
            "accent": "#0f766e",
            "bg": "#fff4f6",
            "surface": "#fffafb",
            "surface_alt": "#ffe3ea",
            "text": "#2f1d25",
            "muted": "#7e6170",
            "hero_start": "#3f1a2b",
            "hero_end": "#8f2145",
            "border": "rgba(47, 29, 37, 0.14)",
        },
    }
    if mood_override and mood_override in palettes:
        palette = dict(palettes[mood_override])
    else:
        palette = dict(palettes.get(mode, palettes["warm"]))
    if high_contrast:
        palette["primary"] = "#9a3412" if mode == "warm" else "#fbbf24"
        palette["accent"] = "#115e59" if mode == "warm" else "#22d3ee"
        palette["border"] = "rgba(0, 0, 0, 0.45)" if mode == "warm" else "rgba(255, 255, 255, 0.45)"
        palette["muted"] = palette["text"]
    return palette


def inject_custom_css(preferences):
    palette = theme_palette(preferences, mood=st.session_state.get("ui_mood", "neutral"))
    font_scale = max(0.85, min(1.35, float(preferences.get("font_scale", 1.0))))
    stylesheet = load_ui_stylesheet()
    st.markdown(
        f"""
        <style>
        :root {{
            --dad-primary: {palette["primary"]};
            --dad-accent: {palette["accent"]};
            --dad-bg: {palette["bg"]};
            --dad-surface: {palette["surface"]};
            --dad-surface-alt: {palette["surface_alt"]};
            --dad-text: {palette["text"]};
            --dad-muted: {palette["muted"]};
            --dad-border: {palette["border"]};
            --dad-hero-start: {palette["hero_start"]};
            --dad-hero-end: {palette["hero_end"]};
            --dad-font-scale: {font_scale};
            --dad-shadow-soft: 0 12px 28px rgba(15, 23, 42, 0.10);
            --dad-shadow-card: 0 16px 32px rgba(15, 23, 42, 0.14);
            --dad-button-text: #f8fafc;
        }}
        {stylesheet}
        </style>
        """,
        unsafe_allow_html=True,
    )


def inject_pwa_metadata(preferences):
    palette = theme_palette(preferences, mood=st.session_state.get("ui_mood", "neutral"))
    manifest_url = pwa_asset_url(PWA_MANIFEST_FILE)
    icon_url = pwa_asset_url(PWA_ICON_FILE)
    theme_color = palette["primary"]
    sync_query = urlencode(
        {
            "manifest": manifest_url,
            "icon": icon_url,
            "theme": theme_color,
            "title": "Dad Bot",
        }
    )
    sync_url = f"{pwa_asset_url(PWA_HEAD_SYNC_FILE)}?{sync_query}"

    iframe = getattr(st, "iframe", None)
    if callable(iframe):
        iframe(sync_url, height=1, width=1)
    else:
        components.iframe(sync_url, height=1, width=1)


def render_hero(bot: "DadBot", active_thread: dict):
    """Warm, personal, emotionally grounded hero banner."""
    shell = ui_shell_snapshot(bot)
    ollama_status = shell.get("ollama", {})
    connected = bool(ollama_status.get("connected", False))
    mood = state_manager.current_ui_mood(default=str(get_chat_event_api().last_saved_mood() or "neutral")).title()

    indicator_color = "#4ade80" if connected else "#f87171"
    indicator_text = "Online" if connected else "Offline"

    st.markdown(
        f"""
        <div class="hero" style="text-align:center; padding: 1.5rem 1.2rem; border-radius: 22px; box-shadow: 0 12px 28px rgba(0,0,0,0.12);">
            <h1 style="margin:0; font-size:2rem; font-weight:600; letter-spacing:-0.02em;">
                Hey Tony 👋
            </h1>
            <p style="margin:0.55rem 0 0; font-size:1rem; opacity:0.96; line-height:1.4;">
                I'm right here, buddy. Always in your corner.
            </p>
            <p style="margin:0.75rem 0 0; font-size:0.92rem; opacity:0.9;">
                <span style="display:inline-block; width:11px; height:11px; border-radius:50%; background:{indicator_color}; vertical-align:middle; margin-right:8px;"></span>
                Ollama {indicator_text} • Mood: <strong>{mood}</strong><br>
                Thread: <strong>{active_thread.get("title", "General Chat")}</strong>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_status_strip(bot: "DadBot"):
    api = get_chat_event_api()
    profile_runtime = getattr(api, "profile_runtime", None)
    living_snapshot = getattr(profile_runtime, "living_dad_snapshot", None)
    living_raw = living_snapshot(limit=2) if callable(living_snapshot) else {"counts": {"proactive_queue": 0}}
    living = dict(living_raw) if isinstance(living_raw, dict) else {"counts": {"proactive_queue": 0}}
    counts = dict(living.get("counts") or {})
    cards = [
        (str(len(api.list_chat_threads())), "Threads"),
        (str(len(api.reminder_catalog())), "Open reminders"),
        (str(int(counts.get("proactive_queue", 0) or 0)), "Queued check-ins"),
    ]
    card_markup = "".join(
        f'<div class="status-card"><strong>{value}</strong><span>{label}</span></div>' for value, label in cards
    )
    st.markdown(f'<div class="status-strip">{card_markup}</div>', unsafe_allow_html=True)


def render_presence_layer(bot: "DadBot", active_thread: dict) -> None:
    api = get_chat_event_api()
    shell = ui_shell_snapshot(bot)
    ollama_status = dict(shell.get("ollama", {}) or {})
    connected = bool(ollama_status.get("connected", False))
    mood = state_manager.current_ui_mood(default=str(api.last_saved_mood() or "neutral")).title()
    thread_title = str(active_thread.get("title") or "Chat")

    col1, col2 = st.columns([1, 7])
    with col1:
        if api.current_avatar_exists():
            try:
                st.image(str(api.avatar_path()), width=56)
            except Exception:
                st.caption("👨")
        else:
            st.caption("👨")
    with col2:
        indicator = "🟢" if connected else "🔴"
        st.caption(f"{indicator} Dad is {'online' if connected else 'offline'}")
        st.caption(f"Presence: {mood} | Thread: {thread_title}")


def render_live_step_graph(*, thread_id: str) -> None:
    timeline = [
        item
        for item in runtime_turn_timeline()
        if str(item.get("thread_id") or "") == str(thread_id or "default")
    ]
    if not timeline:
        return

    events = [str(item.get("event_type") or "event") for item in timeline[-6:]]
    label_map = {
        "user_message": "User",
        "thinking_update": "Think",
        "decision_event": "Decide",
        "photo_request": "Photo",
        "tts_request": "TTS",
        "assistant_reply": "Reply",
        "guardrail_rejection": "Guard",
    }
    chips = "".join(
        f"<span class='dad-step-chip'>{label_map.get(event, event.replace('_', ' ').title())}</span>"
        for event in events
    )
    st.markdown(f"<div class='dad-step-graph'>{chips}</div>", unsafe_allow_html=True)


def _fork_messages_from_visible(visible_messages: list[dict], *, through_index: int) -> list[dict]:
    forked: list[dict] = []
    for index, msg in enumerate(visible_messages):
        if index > through_index:
            break
        role = str(msg.get("role") or "").strip().lower()
        if role not in {"user", "assistant"}:
            continue
        forked.append(
            {
                "role": role,
                "content": str(msg.get("content") or "").strip(),
                "attachments": list(msg.get("attachments") or []),
            }
        )
    return [item for item in forked if item.get("content") or item.get("attachments")]


def fork_thread_from_message(*, visible_messages: list[dict], through_index: int, parent_thread_id: str = "") -> str | None:
    api = get_chat_event_api()
    payload = _fork_messages_from_visible(visible_messages, through_index=through_index)
    if not payload:
        return None
    new_thread = api.create_chat_thread()
    new_thread_id = str(new_thread.get("thread_id") or "")
    if not new_thread_id:
        return None
    api.seed_thread(new_thread_id, payload)
    if parent_thread_id:
        meta_map = thread_ui_metadata()
        meta = dict(meta_map.get(new_thread_id) or {})
        meta["parent_thread_id"] = str(parent_thread_id or "")
        meta["fork_from_message_index"] = int(through_index)
        meta_map[new_thread_id] = meta
        st.session_state["thread_ui_meta"] = meta_map
    switch_active_thread(new_thread_id)
    return new_thread_id


def thread_ui_metadata() -> dict[str, dict]:
    return cast(dict[str, dict], st.session_state.setdefault("thread_ui_meta", {}))


def thread_label_with_meta(thread: dict) -> str:
    meta = dict(thread_ui_metadata().get(str(thread.get("thread_id") or "")) or {})
    title = str(meta.get("title") or thread.get("title") or "Chat").strip() or "Chat"
    turns = int(thread.get("turn_count", 0) or 0)
    closed_suffix = " | closed" if thread.get("closed") else ""
    pin_prefix = "📌 " if bool(meta.get("pinned", False)) else ""
    return f"{pin_prefix}{title} | {turns} turns{closed_suffix}"


def queued_message_outbox() -> list[dict]:
    return cast(list[dict], st.session_state.setdefault("runtime_outbox_queue", []))


def _runtime_outbox_disk_payload() -> list[dict]:
    if not RUNTIME_OUTBOX_PATH.exists():
        return []
    try:
        payload = json.loads(RUNTIME_OUTBOX_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(payload, list):
        return []
    return [dict(item) for item in payload if isinstance(item, dict)]


def _persist_runtime_outbox(queue: list[dict]) -> None:
    RUNTIME_OUTBOX_PATH.parent.mkdir(parents=True, exist_ok=True)
    normalized = []
    for item in list(queue or []):
        prompt = str(item.get("prompt") or "").strip()
        if not prompt:
            continue
        normalized.append(
            {
                "thread_id": str(item.get("thread_id") or "default"),
                "prompt": prompt,
                "attachments": list(item.get("attachments") or []),
                "queued_at": float(item.get("queued_at") or time.time()),
                "gateway": dict(item.get("gateway") or {}),
            }
        )
    RUNTIME_OUTBOX_PATH.write_text(json.dumps(normalized[-50:], ensure_ascii=True, indent=2), encoding="utf-8")


def restore_runtime_outbox_once() -> None:
    if bool(st.session_state.get("runtime_outbox_restored", False)):
        return
    persisted = _runtime_outbox_disk_payload()
    if persisted:
        st.session_state["runtime_outbox_queue"] = persisted
    st.session_state["runtime_outbox_restored"] = True


def queue_message_for_retry(
    *,
    thread_id: str,
    prompt: str,
    attachments: list[dict] | None = None,
    gateway: dict | None = None,
) -> None:
    queue = queued_message_outbox()
    queue.append(
        {
            "thread_id": str(thread_id or "default"),
            "prompt": str(prompt or "").strip(),
            "attachments": list(attachments or []),
            "queued_at": time.time(),
            "gateway": dict(gateway or {}),
        }
    )
    st.session_state["runtime_outbox_queue"] = queue[-50:]
    _persist_runtime_outbox(st.session_state["runtime_outbox_queue"])


def flush_queued_messages(*, limit: int = 2) -> tuple[int, int]:
    queue = queued_message_outbox()
    if not queue:
        return 0, 0
    sent = 0
    remaining: list[dict] = []
    for index, item in enumerate(queue):
        if sent >= max(1, int(limit or 1)):
            remaining.extend(queue[index:])
            break
        prompt = str(item.get("prompt") or "").strip()
        if not prompt:
            continue
        result = process_prompt_via_runtime(
            thread_id=str(item.get("thread_id") or "default"),
            prompt=prompt,
            attachments=list(item.get("attachments") or []),
            message_metadata={"gateway": dict(item.get("gateway") or {})}
            if dict(item.get("gateway") or {}) 
            else None,
        )
        if str(result.get("degraded_mode") or "normal") == "normal":
            sent += 1
        else:
            remaining.append(item)
            remaining.extend(queue[index + 1 :])
            break
    st.session_state["runtime_outbox_queue"] = remaining
    _persist_runtime_outbox(remaining)
    return sent, len(remaining)


def clear_queued_messages() -> None:
    st.session_state["runtime_outbox_queue"] = []
    _persist_runtime_outbox([])


def runtime_delivery_snapshot() -> dict:
    metrics = dict(st.session_state.get("runtime_delivery_metrics") or {})
    ttft_series = list(metrics.get("ttft_ms") or [])
    avg_ttft_ms = int(sum(ttft_series) / len(ttft_series)) if ttft_series else 0
    sent = int(metrics.get("sent", 0) or 0)
    dropouts = int(metrics.get("dropouts", 0) or 0)
    dropout_rate = (dropouts / sent) if sent > 0 else 0.0
    return {
        **metrics,
        "avg_ttft_ms": avg_ttft_ms,
        "dropout_rate": dropout_rate,
        "queued": len(queued_message_outbox()),
        "reconnecting": bool(st.session_state.get("runtime_reconnecting", False)),
    }


def reliability_state_from_snapshot(snapshot: dict) -> str:
    if bool(snapshot.get("reconnecting", False)):
        return "retrying"
    if int(snapshot.get("queued", 0) or 0) > 0:
        return "replay_pending"
    if float(snapshot.get("dropout_rate", 0.0) or 0.0) >= 0.2:
        return "degraded"
    return "healthy"


def apply_prompt_controls(*, thread_id: str, prompt: str) -> tuple[str, str, list[str]]:
    scope = str(st.session_state.get("agent_control_scope", "thread") or "thread").strip().lower()
    scope_key = "global" if scope == "global" else str(thread_id or "default")
    mode_map = cast(dict[str, str], st.session_state.setdefault("agent_mode_by_scope", {}))
    permissions_map = cast(dict[str, list[str]], st.session_state.setdefault("tool_permissions_by_scope", {}))
    mode = str(mode_map.get(scope_key) or "balanced")
    allowed_tools = list(permissions_map.get(scope_key) or ["memory_lookup", "photo_generation", "tts"])
    blocked_tools = [
        tool
        for tool in ["memory_lookup", "web_lookup", "photo_generation", "tts", "document_read"]
        if tool not in set(allowed_tools)
    ]
    instruction = (
        f"[Agent mode: {mode}. Allowed tools: {', '.join(allowed_tools) or 'none'}. "
        f"Blocked tools: {', '.join(blocked_tools) or 'none'}.]"
    )
    return f"{instruction}\n\n{str(prompt or '').strip()}", mode, allowed_tools


def apply_gateway_ingress_context(*, thread_id: str, prompt: str) -> tuple[str, dict]:
    pending_by_thread = cast(dict[str, dict], st.session_state.setdefault("gateway_ingress_by_thread", {}))
    context = dict(pending_by_thread.pop(str(thread_id or "default"), {}) or {})
    st.session_state["gateway_ingress_by_thread"] = pending_by_thread
    if not context:
        return str(prompt or "").strip(), {}
    channel = str(context.get("channel") or "chat").strip().lower() or "chat"
    sender = str(context.get("sender") or "unknown").strip() or "unknown"
    sender_name = str(context.get("sender_name") or "").strip()
    message_id = str(context.get("message_id") or "").strip()
    conversation_id = str(context.get("conversation_id") or "").strip()
    delivery_mode = str(context.get("delivery_mode") or "sync").strip().lower() or "sync"
    envelope_lines = [
        f"channel={channel}",
        f"sender={sender}",
        f"delivery_mode={delivery_mode}",
    ]
    if sender_name:
        envelope_lines.append(f"sender_name={sender_name}")
    if message_id:
        envelope_lines.append(f"message_id={message_id}")
    if conversation_id:
        envelope_lines.append(f"conversation_id={conversation_id}")
    envelope = "[Gateway ingress: " + ", ".join(envelope_lines) + "]"
    return f"{envelope}\n\n{str(prompt or '').strip()}", context


def run_frontier_heartbeat(bot: "DadBot", *, force: bool = True) -> dict:
    scheduler = getattr(bot, "maintenance_scheduler", None)
    if scheduler is None or not callable(getattr(scheduler, "run_proactive_heartbeat", None)):
        return {"ok": False, "error": "maintenance scheduler unavailable"}
    try:
        result = dict(scheduler.run_proactive_heartbeat(force=force) or {})
        return {"ok": True, "result": result}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def run_frontier_self_improvement(bot: "DadBot", *, background: bool = False) -> dict:
    manager = getattr(bot, "long_term_signals", None)
    if manager is None:
        return {"ok": False, "error": "long_term_signals unavailable"}
    try:
        if background and callable(getattr(manager, "schedule_continuous_learning", None)):
            task = manager.schedule_continuous_learning()
            task_id = str(getattr(task, "dadbot_task_id", "") or "") if task is not None else ""
            return {"ok": True, "queued": bool(task_id), "task_id": task_id}
        runner = getattr(manager, "perform_continuous_learning_cycle", None)
        if not callable(runner):
            return {"ok": False, "error": "continuous learning runner unavailable"}
        result_payload = runner()
        normalized_result = dict(result_payload) if isinstance(result_payload, dict) else {"value": result_payload}
        return {"ok": True, "result": normalized_result}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def run_frontier_browser_open(url: str) -> dict:
    target = str(url or "").strip()
    if not target:
        return {"ok": False, "error": "url is required"}
    if not re.match(r"^(https?|file)://", target, re.IGNORECASE):
        return {"ok": False, "error": "url must start with http://, https://, or file://"}
    try:
        opened = bool(webbrowser.open(target, new=2))
        return {"ok": True, "opened": opened, "url": target}
    except Exception as exc:
        return {"ok": False, "error": str(exc), "url": target}


def evaluate_tool_policy_violation(prompt: str, allowed_tools: list[str]) -> str:
    text = str(prompt or "").strip().lower()
    allowed = {str(item).strip().lower() for item in list(allowed_tools or [])}
    checks = [
        (
            "photo_generation",
            ["photo", "image", "picture", "selfie"],
            "Tool policy blocked this turn: photo_generation is disabled for current scope.",
        ),
        (
            "web_lookup",
            ["search web", "look up", "google", "website", "http://", "https://", "browse"],
            "Tool policy blocked this turn: web_lookup is disabled for current scope.",
        ),
        (
            "tts",
            ["read aloud", "say this", "voice output", "speak this"],
            "Tool policy blocked this turn: tts is disabled for current scope.",
        ),
        (
            "document_read",
            [
                "read this file",
                "read file",
                "open file",
                "load file",
                "read document",
                "open document",
                "read pdf",
                ".pdf",
            ],
            "Tool policy blocked this turn: document_read is disabled for current scope.",
        ),
        (
            "memory_lookup",
            ["remember", "what do you recall", "from our history"],
            "Tool policy blocked this turn: memory_lookup is disabled for current scope.",
        ),
    ]
    for capability, keywords, message in checks:
        if capability in allowed:
            continue
        if any(keyword in text for keyword in keywords):
            return message
    return ""


def render_thread_branch_map(*, active_thread_id: str) -> None:
    api = get_chat_event_api()
    threads = list(api.list_chat_threads() or [])
    meta = thread_ui_metadata()
    parent_to_children: dict[str, list[str]] = {}
    title_map: dict[str, str] = {}
    for thread in threads:
        thread_id = str(thread.get("thread_id") or "")
        if not thread_id:
            continue
        title_map[thread_id] = str(dict(meta.get(thread_id) or {}).get("title") or thread.get("title") or thread_id)
        parent = str(dict(meta.get(thread_id) or {}).get("parent_thread_id") or "").strip()
        if parent:
            parent_to_children.setdefault(parent, []).append(thread_id)
    if not parent_to_children:
        st.caption("No thread forks yet. Use 'Fork from here' on any message to create a branch.")
        return
    child_ids = {child for children in parent_to_children.values() for child in children}
    roots = [thread_id for thread_id in title_map if thread_id not in child_ids]
    roots = roots or list(title_map.keys())
    for root in roots[:12]:
        st.markdown(f"- {'**' if root == active_thread_id else ''}{title_map.get(root, root)}{'**' if root == active_thread_id else ''}")
        for child in parent_to_children.get(root, [])[:12]:
            child_meta = dict(meta.get(child) or {})
            index_marker = child_meta.get("fork_from_message_index")
            suffix = f" (fork @ msg {index_marker})" if isinstance(index_marker, int) else ""
            st.caption(f"  -> {title_map.get(child, child)}{suffix}")


def try_gateway_ingest_mirror(*, channel: str, thread_id: str, prompt: str, gateway: dict) -> dict:
    if not bool(st.session_state.get("frontier_use_api_ingest", False)):
        return {"mirrored": False, "reason": "disabled"}
    base_url = str(
        st.session_state.get("frontier_gateway_base_url")
        or os.environ.get("DADBOT_SERVICE_URL")
        or "http://127.0.0.1:8010"
    ).strip().rstrip("/")
    normalized_channel = re.sub(r"[^a-z0-9._-]+", "-", str(channel or "chat").strip().lower()).strip("-._") or "chat"
    payload = {
        "message": str(prompt or "").strip(),
        "tenant_id": str(get_chat_event_api().tenant_id() or "default"),
        "sender_id": str(gateway.get("sender") or "user-1"),
        "sender_name": str(gateway.get("sender_name") or ""),
        "external_message_id": str(gateway.get("message_id") or ""),
        "conversation_id": str(gateway.get("conversation_id") or ""),
    }
    url = f"{base_url}/channels/{normalized_channel}/sessions/{thread_id}/ingest"
    try:
        req = urllib_request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib_request.urlopen(req, timeout=2.5) as response:
            body = response.read().decode("utf-8", errors="replace")
        return {"mirrored": True, "url": url, "response": body[:180]}
    except urllib_error.URLError as exc:
        return {"mirrored": False, "url": url, "reason": str(exc)}
    except Exception as exc:
        return {"mirrored": False, "url": url, "reason": str(exc)}


def render_inline_citations(message: dict) -> None:
    content = str(message.get("content") or "")
    urls = sorted(set(re.findall(r"https?://[^\s)\]>]+", content)))
    source_mentions = sorted(set(re.findall(r"Source:\s*([^\.\n]+)", content)))
    if not urls and not source_mentions:
        return
    with st.expander("Sources", expanded=False):
        for source in source_mentions[:5]:
            st.caption(f"- {source.strip()}")
        for url in urls[:6]:
            st.caption(f"- {url.strip()}")


def delete_message_from_thread(*, thread_id: str, visible_messages: list[dict], msg_index: int) -> bool:
    api = get_chat_event_api()
    payload = [
        {
            "role": str(msg.get("role") or ""),
            "content": str(msg.get("content") or ""),
            "attachments": list(msg.get("attachments") or []),
        }
        for idx, msg in enumerate(visible_messages)
        if idx != int(msg_index)
    ]
    payload = [entry for entry in payload if entry.get("role") in {"user", "assistant"}]
    if not payload:
        payload = default_thread_messages()
    api.seed_thread(str(thread_id or api.active_thread_id or "default"), payload)
    return True


def inject_full_chat_mode_css() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(900px 460px at -8% -10%, color-mix(in srgb, var(--dad-accent) 14%, transparent), transparent 58%),
                radial-gradient(860px 420px at 112% -6%, color-mix(in srgb, var(--dad-primary) 18%, transparent), transparent 60%),
                linear-gradient(180deg, color-mix(in srgb, var(--dad-bg) 90%, white), var(--dad-bg));
        }

        .main .block-container {
            max-width: 1040px;
            padding-top: 0.9rem;
            padding-bottom: 2.8rem;
        }

        .dad-full-chat-topbar {
            position: sticky;
            top: 0.4rem;
            z-index: 900;
            border: 1px solid var(--dad-border);
            border-radius: 18px;
            background: color-mix(in srgb, var(--dad-surface) 84%, transparent);
            backdrop-filter: blur(10px);
            box-shadow: 0 14px 28px rgba(15, 23, 42, 0.14);
            margin-bottom: 0.95rem;
            padding: 0.45rem 0.55rem;
        }

        .hero {
            border-radius: 28px;
            padding: 2.1rem 1.8rem;
            box-shadow: 0 22px 44px rgba(15, 23, 42, 0.3);
            margin-bottom: 1rem;
        }

        .hero h1 {
            font-size: clamp(1.8rem, 2.8vw, 2.3rem);
            letter-spacing: -0.01em;
        }

        .status-strip {
            gap: 0.95rem;
            margin: 1rem 0 1.25rem;
        }

        .status-card {
            border-radius: 18px;
            padding: 0.95rem 1rem;
            box-shadow: 0 14px 30px rgba(15, 23, 42, 0.1);
        }

        .dad-floating-actions {
            border-radius: 20px;
            padding: 0.65rem;
            border: 1px solid color-mix(in srgb, var(--dad-border) 78%, white);
            background: linear-gradient(
                140deg,
                color-mix(in srgb, var(--dad-surface) 91%, transparent),
                color-mix(in srgb, var(--dad-surface-alt) 36%, var(--dad-surface))
            );
            box-shadow: 0 16px 30px rgba(15, 23, 42, 0.12);
            margin-top: 0.45rem;
        }

        div[data-testid="stChatMessage"] {
            border-radius: 20px;
            padding: 18px 20px;
        }

        div[data-testid="stChatInput"] {
            border-radius: 16px;
            box-shadow: 0 16px 34px rgba(15, 23, 42, 0.16);
            border: 1px solid color-mix(in srgb, var(--dad-border) 78%, white);
        }

        .stButton > button {
            border-radius: 12px;
            letter-spacing: 0.01em;
        }

        @media (max-width: 768px) {
            .main .block-container {
                max-width: 100%;
                padding-left: 0.85rem;
                padding-right: 0.85rem;
            }

            .dad-full-chat-topbar {
                top: 0.3rem;
                border-radius: 14px;
                margin-bottom: 0.8rem;
            }

            .hero {
                border-radius: 20px;
                padding: 1.3rem 1rem;
            }

            .status-strip {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_mobile_tab(bot: "DadBot"):
    api = get_chat_event_api()
    preferences = ui_preferences()
    active_thread = api.active_chat_thread() or {}
    assets = pwa_asset_status()
    assets_ready = all(assets.values())
    vision_ready, vision_message = api.vision_fallback_status()
    thread_labels = {}
    current_selection = None
    for thread in api.list_chat_threads():
        label = f"{thread.get('title', 'Chat')} ({thread.get('turn_count', 0)} turns)"
        if thread.get("closed"):
            label = f"{label} - closed"
        thread_labels[label] = thread.get("thread_id")
        if thread.get("thread_id") == api.active_thread_id:
            current_selection = label

    st.subheader("Mobile app shell")
    st.caption("Phone-first install and quick-action surface for running Dad Bot as a homescreen companion.")

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("Install assets", "Ready" if assets_ready else "Missing")
    metric_col2.metric("Photo support", "Ready" if vision_ready else "Fallback")
    metric_col3.metric("Theme", titleize_token(preferences.get("theme_mode"), "Warm"))

    st.markdown(
        """
        <div class="mobile-card install-note">
            <h3>Install path</h3>
            <p>
                The app now exposes a real web manifest and mobile meta tags through Streamlit static serving.
                Browser install behavior is strongest on HTTPS or localhost. Full offline shell caching still needs a root-scoped frontend or proxy-served service worker.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    install_col1, install_col2 = st.columns(2)
    with install_col1:
        st.markdown(
            """
            <div class="mobile-card">
                <h3>Android / Chromium</h3>
                <ol>
                    <li>Open the app in Chrome or Edge.</li>
                    <li>Use the browser menu and choose Install app or Add to Home screen.</li>
                    <li>Pin the app and keep the Dad Bot tab selected for fastest return.</li>
                </ol>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with install_col2:
        st.markdown(
            """
            <div class="mobile-card">
                <h3>iPhone / Safari</h3>
                <ol>
                    <li>Open the app in Safari.</li>
                    <li>Tap Share, then Add to Home Screen.</li>
                    <li>Launch from the homescreen for the cleanest full-screen shell.</li>
                </ol>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.container(border=True):
        st.subheader("Quick mobile actions")
        action_col1, action_col2 = st.columns(2)
        with action_col1:
            if st.button("Start Mobile Thread", use_container_width=True, type="primary"):
                create_new_thread()
                st.rerun()
        with action_col2:
            if st.button("Send Photo to Thread", use_container_width=True):
                if emit_generated_photo_message(bot, str(api.active_thread_id or "default")):
                    st.rerun()

        selected_thread_label = (
            st.selectbox(
                "Jump to thread without opening the sidebar",
                options=list(thread_labels.keys()),
                index=list(thread_labels.keys()).index(current_selection) if current_selection in thread_labels else 0,
            )
            if thread_labels
            else None
        )
        selected_thread_id = thread_labels.get(selected_thread_label) if selected_thread_label else None
        if st.button(
            "Switch to Selected Thread",
            use_container_width=True,
            disabled=not selected_thread_id or selected_thread_id == api.active_thread_id,
        ):
            switch_active_thread(str(selected_thread_id))
            st.rerun()

    with st.container(border=True):
        st.subheader("Readiness")
        st.write(
            {
                "active_thread": active_thread.get("title", "Chat"),
                "active_thread_closed": bool(active_thread.get("closed")),
                "manifest": pwa_asset_url(PWA_MANIFEST_FILE) if assets["manifest"] else "missing",
                "icon": pwa_asset_url(PWA_ICON_FILE) if assets["icon"] else "missing",
                "maskable_icon": pwa_asset_url(PWA_MASKABLE_ICON_FILE) if assets["maskable_icon"] else "missing",
                "vision": vision_message,
                "light_mode": bool(preferences.get("light_mode")),
                "high_contrast": bool(preferences.get("high_contrast")),
            }
        )
        st.markdown(
            "<div class='mobile-chip-row'>"
            f"<span class='mobile-chip'>Tabs wrap cleanly on small screens</span>"
            f"<span class='mobile-chip'>Theme color follows {titleize_token(preferences.get('theme_mode'), 'Warm')}</span>"
            f"<span class='mobile-chip'>Static assets served from /app/static</span>"
            "</div>",
            unsafe_allow_html=True,
        )


def update_ui_mood(bot: "DadBot"):
    _ = bot
    state_manager.set_ui_mood(str(get_chat_event_api().last_saved_mood() or "neutral"))


def optimize_runtime_for_hardware(bot: "DadBot"):
    api = get_chat_event_api()
    cpu_count = max(1, int(os.cpu_count() or 1))
    low_power = cpu_count <= 4
    medium_power = 4 < cpu_count <= 8
    if low_power:
        light_mode = True
        stream_max_chars = 8000
        max_thinking_time = 30
    elif medium_power:
        light_mode = True
        stream_max_chars = 12000
        max_thinking_time = 45
    else:
        light_mode = False
        stream_max_chars = 16000
        max_thinking_time = 60

    preferences = ui_preferences()
    preferences["light_mode"] = light_mode
    apply_ui_preferences(cast("DadBot", api))
    api.update_runtime_profile(
        {
            "stream_max_chars": stream_max_chars,
            "max_thinking_time_seconds": max_thinking_time,
        },
        save=False,
    )
    api.save_profile()
    return {
        "cpu_count": cpu_count,
        "light_mode": light_mode,
        "stream_max_chars": stream_max_chars,
        "max_thinking_time_seconds": max_thinking_time,
    }


def default_thread_messages():
    return [{"role": "assistant", "content": get_chat_event_api().opening_message(INITIAL_GREETING)}]


def bot_messages_for_thread(thread_id: str):
    return get_chat_event_api().snapshot_thread_messages(thread_id, default_greeting=INITIAL_GREETING)


def initialize_session(bot: "DadBot"):
    api = get_chat_event_api()
    restore_runtime_outbox_once()
    api.ensure_chat_thread_state()
    api.sync_active_thread_snapshot()
    if "active_thread_id" not in st.session_state:
        st.session_state.active_thread_id = api.active_thread_id
    if "export_path" not in st.session_state:
        st.session_state.export_path = str(DEFAULT_EXPORT_PATH)
    if st.session_state.active_thread_id != api.active_thread_id:
        api.switch_chat_thread(st.session_state.active_thread_id)
    if not st.session_state.get("ui_profile_seeded", False):
        sync_ui_voice_from_profile(bot)
        st.session_state.ui_profile_seeded = True


def switch_active_thread(thread_id: str):
    api = get_chat_event_api()
    api.switch_chat_thread(thread_id)
    st.session_state.active_thread_id = api.active_thread_id


def create_new_thread():
    api = get_chat_event_api()
    thread = api.create_chat_thread()
    st.session_state.active_thread_id = thread["thread_id"]
    api.seed_thread(thread["thread_id"], default_thread_messages())
    return thread


def first_run_wizard(bot: "DadBot"):
    """One-time onboarding wizard – runs only on the very first launch.

    Detected via 'onboarding_complete' in session_state. Once the user
    clicks Finish, the flag is persisted to the profile so it never shows again.
    """

    try:
        st.markdown("<div style='background:#ffecb3;padding:6px 0;text-align:center;font-weight:bold;'>[DEBUG] first_run_wizard() entered</div>", unsafe_allow_html=True)
        # Skip if already completed this session or previously persisted
        api = get_chat_event_api()
        if st.session_state.get("onboarding_complete", False):
            return
        if api.onboarding_complete():
            st.session_state["onboarding_complete"] = True
            return

        st.title("👋 Welcome to Dad Bot!")
        st.caption("Let's set up your personal digital dad in 60 seconds.")

        step = st.session_state.get("onboarding_step", 0)

        if step == 0:
            st.markdown("### Hi! I'm Dad.")
            st.write(
                "I'll remember everything you share with me, speak like a real dad, "
                "and always have your back. This takes about a minute."
            )
            if st.button("Let's set you up →", type="primary", use_container_width=True):
                st.session_state["onboarding_step"] = 1
                st.rerun()

        elif step == 1:
            st.subheader("Step 1 of 4 — Choose your brain")
            st.caption("Pick the Ollama model Dad will think with.")
            model = st.selectbox(
                "Main language model",
                ["llama3.2", "llama3.3", "qwen2.5:14b", "gemma2:27b"],
                index=0,
                help="llama3.2 is fast and works well on most machines. Larger models give richer responses.",
            )
            if st.button("Next →", type="primary", use_container_width=True):
                api.set_profile_model(model)
                st.session_state["onboarding_step"] = 2
                st.rerun()

        elif step == 2:
            st.subheader("Step 2 of 4 — Voice settings")
            st.caption("How should Dad speak to you?")
            voice_backend = st.radio(
                "Text-to-speech engine",
                ["pyttsx3 (lightweight, works out of the box)", "Piper (natural neural TTS, requires setup)"],
                index=0,
            )
            backend_key = "piper" if "Piper" in voice_backend else "pyttsx3"
            if backend_key == "piper":
                st.info(
                    "Piper requires the `piper` binary and an `.onnx` model file. See Preferences → Voice to configure after setup."
                )
            if st.button("Next →", type="primary", use_container_width=True):
                api.update_voice_profile({"tts_backend": backend_key})
                st.session_state["onboarding_step"] = 3
                st.rerun()

        elif step == 3:
            st.subheader("Step 3 of 4 — Create Dad's face")
            st.caption("Generate a custom avatar using an Ollama image model (optional).")
            custom_prompt = st.text_area(
                "Avatar description",
                value="Photorealistic warm portrait of a friendly 56-year-old father with kind eyes, "
                "flannel shirt, cozy kitchen background, cinematic lighting",
            )
            _gen_col, _skip_col = st.columns(2)
            if _gen_col.button("Generate Avatar", type="primary", use_container_width=True):
                with st.spinner("Dad is getting his picture taken..."):
                    ok = api.generate_avatar(custom_prompt or None)
                if ok:
                    st.success("Avatar created!")
                    if api.current_avatar_exists():
                        st.image(str(api.avatar_path()), width=240)
                else:
                    st.warning(
                        "Could not generate avatar right now – using emoji fallback. You can try again from Preferences."
                    )
                st.session_state["onboarding_step"] = 4
                st.rerun()
            if _skip_col.button("Skip →", use_container_width=True):
                st.session_state["onboarding_step"] = 4
                st.rerun()

        elif step >= 4:
            st.subheader("Step 4 of 4 — Connect your calendar (optional)")
            st.caption("Paste a public .ics feed so Dad knows about your upcoming events.")
            ical = st.text_input(
                "Public iCal feed URL (.ics)",
                placeholder="https://calendar.google.com/calendar/ical/.../basic.ics",
                help="In Google Calendar: Settings → your calendar → Integrate → Public address in iCal format.",
            )
            if ical:
                api.set_ical_feed_url(ical, save=False)
                st.success("Calendar feed URL noted – will be saved when you finish.")
            if st.button("Finish Setup →", type="primary", use_container_width=True):
                api.set_onboarding_complete(True)
                api.save_profile()
                st.session_state["onboarding_complete"] = True
                st.balloons()
                st.rerun()
            if st.button("Skip →", use_container_width=True):
                api.set_onboarding_complete(True)
                api.save_profile()
                st.session_state["onboarding_complete"] = True
                st.rerun()

        return
    except Exception as e:
        st.error(f"Wizard crashed: {e}")
        import traceback
        st.code(traceback.format_exc())
        return


def require_pin(bot: "DadBot"):
    api = get_chat_event_api()
    security = api.streamlit_security_settings()
    if security is None:
        security = {}
    if not security.get("require_pin"):
        return
    if st.session_state.get("access_granted"):
        return
    st.title("Dad Bot")
    st.caption("This workspace is PIN protected.")
    if security.get("pin_hint"):
        st.info(f"Hint: {security['pin_hint']}")
    pin = st.text_input("Enter PIN", type="password")
    if st.button("Unlock", type="primary"):
        if api.verify_streamlit_pin(pin):
            st.session_state.access_granted = True
            st.rerun()
        else:
            st.error("Incorrect PIN")
    return


def _create_placeholder_dad_image() -> bytes:
    """Create a simple placeholder image when Ollama models aren't available."""
    try:
        # Create a warm, welcoming image with a color gradient
        img = Image.new("RGB", (400, 500), color=(220, 180, 140))  # Warm beige background
        draw = ImageDraw.Draw(img)

        # Try to use a default font, fall back to default if unavailable
        try:
            title_font = ImageFont.truetype("arial.ttf", 36)
            subtitle_font = ImageFont.truetype("arial.ttf", 20)
        except OSError:
            title_font = ImageFont.load_default()
            subtitle_font = ImageFont.load_default()

        # Draw a simple face representation
        # Head circle
        draw.ellipse([(80, 80), (320, 320)], fill=(240, 200, 160), outline=(200, 140, 80), width=3)

        # Eyes
        draw.ellipse([(120, 140), (160, 180)], fill=(100, 100, 100))  # Left eye
        draw.ellipse([(240, 140), (280, 180)], fill=(100, 100, 100))  # Right eye
        draw.ellipse([(130, 150), (150, 170)], fill=(255, 255, 255))  # Left pupil highlight
        draw.ellipse([(250, 150), (270, 170)], fill=(255, 255, 255))  # Right pupil highlight

        # Smile (arc-like)
        draw.arc([(140, 180), (260, 260)], 0, 180, fill=(100, 80, 60), width=4)

        # Add text below
        text = "Dad's Photo"
        bbox = draw.textbbox((0, 0), text, font=title_font)
        text_width = bbox[2] - bbox[0]
        draw.text(((400 - text_width) // 2, 360), text, fill=(80, 60, 40), font=title_font)

        subtitle = "(Ollama model needed for AI photo)"
        bbox = draw.textbbox((0, 0), subtitle, font=subtitle_font)
        text_width = bbox[2] - bbox[0]
        draw.text(((400 - text_width) // 2, 420), subtitle, fill=(120, 100, 80), font=subtitle_font)

        # Convert to bytes
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        logger.exception("Failed to create placeholder image")
        # Return a minimal valid 1x1 PNG as final fallback
        return base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
        )


def generate_dad_photo():
    with st.spinner("Dad is taking a quick photo for you..."):
        try:
            candidates = ["flux", "flux-dev", "flux-schnell", "sdxl", "stable-diffusion"]
            model = find_available_image_model(tuple(candidates))
            try:
                import ollama
            except ImportError:
                ollama = None
            if not model or ollama is None:
                st.info("📷 Using fallback image (install Ollama and pull 'flux' or 'sdxl' for AI-generated photos)")
                return _create_placeholder_dad_image()
            response = ollama.generate(
                model=model,
                prompt=(
                    "Photorealistic warm portrait of a friendly 56-year-old father with kind eyes, "
                    "short neatly trimmed graying hair, gentle reassuring smile, wearing a soft flannel shirt, "
                    "standing in a cozy home kitchen with wooden cabinets and soft natural window light, "
                    "heartwarming atmosphere, high detail, cinematic lighting, shot on 50mm lens, 8k"
                ),
                options={"num_predict": 1},
            )
            images = [base64.b64decode(img) for img in response.get("images", []) if img]
            if images:
                return images[0]
            st.warning("Image model returned no images. Try again or switch to a different model.")
            return None
        except Exception as exc:
            logger.exception("Photo generation failed")
            st.error(f"Couldn't generate photo right now: {exc}")
            return None


def emit_generated_photo_message(
    bot: "DadBot", thread_id: str, message: str = "Here's a quick photo I took for you, buddy. Love you."
) -> bool:
    photo = generate_dad_photo()
    if not photo:
        return False
    api = get_chat_event_api()
    api.emit_assistant_photo_message(
        thread_id=str(thread_id or api.active_thread_id or "default"),
        text=message,
        attachment={"type": "image", "data_b64": base64.b64encode(photo).decode()},
    )
    api.process_until_idle(max_events=8)
    return True


def render_companion_toolbar(bot: "DadBot"):
    api = get_chat_event_api()
    st.markdown(
        """
        <div class="dad-companion-toolbar">
            <div>
                <strong>Companion Mode</strong>
                <span>Quick actions for everyday check-ins.</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    action_col1, action_col2, action_col3, action_col4 = st.columns(4)
    if action_col1.button("New Chat", key="companion-new-chat", use_container_width=True):
        create_new_thread()
        st.rerun()
    if action_col2.button("Daily Check-in", key="companion-daily-checkin", use_container_width=True):
        st.session_state.nudge_prompt = "Hey Dad, can we do a quick check-in today?"
        st.rerun()
    if action_col3.button("Mood Lift", key="companion-mood-lift", use_container_width=True):
        st.session_state.nudge_prompt = "Hey Dad, can you give me a little encouragement right now?"
        st.rerun()
    if action_col4.button("Send Photo", key="companion-send-photo", use_container_width=True):
        if emit_generated_photo_message(bot, str(api.active_thread_id or "default")):
            st.rerun()


def render_chat_tab(bot: "DadBot", active_thread: dict):
    initialize_session(bot)
    api = get_chat_event_api()
    hud_mode = str(st.session_state.get("chat_hud_mode", "balanced") or "balanced").strip().lower()
    if hud_mode not in {"minimal", "balanced", "debug"}:
        hud_mode = "balanced"
        st.session_state["chat_hud_mode"] = hud_mode
    if hud_mode == "debug":
        # Debug output gated by flag (add st.markdown here if needed)
        render_runtime_guardrails_card(dismiss_key="dismiss-runtime-guardrails-chat")
        render_runtime_semantic_strip(thread_id=str(api.active_thread_id or "default"))
    active_thread_id = str(api.active_thread_id or "default")
    active_meta = dict(thread_ui_metadata().get(active_thread_id) or {})
    if active_meta.get("title"):
        active_thread = {**dict(active_thread or {}), "title": str(active_meta.get("title") or "")}
    thread_messages = list(
        load_thread_projection(
            api=api,
            thread_id=active_thread_id,
            seed_messages=bot_messages_for_thread(active_thread_id),
        ).messages
        or []
    )
    render_hero(bot, active_thread)
    render_presence_layer(bot, active_thread)
    if hud_mode in {"balanced", "debug"}:
        render_status_strip(bot)
        render_live_step_graph(thread_id=active_thread_id)
    compact_mode = bool(st.session_state.get("chat_compact_mode", True))
    st.markdown("<div class='dad-floating-actions'>", unsafe_allow_html=True)
    action_col1, action_col2, action_col3, action_col4 = st.columns(4)
    if action_col1.button("🧵 New Thread", key="chat-fab-new-thread", use_container_width=True):
        create_new_thread()
        st.rerun()
    if action_col2.button("🎙️ Talk", key="chat-fab-voice", use_container_width=True):
        st.session_state["voice_panel_open"] = True
        st.rerun()
    if action_col3.button("📷 Send Photo", key="chat-fab-photo", use_container_width=True):
        if emit_generated_photo_message(bot, str(api.active_thread_id or "default")):
            st.rerun()
    if action_col4.button("💚 Check-in", key="chat-fab-checkin", use_container_width=True):
        st.session_state.nudge_prompt = "Hey Dad, can we do a quick check-in today?"
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)
    toggle_label = "Expand chat tools" if compact_mode else "Compact chat tools"
    toggle_col1, toggle_col2, toggle_col3 = st.columns([1.4, 3.2, 2.4])
    if toggle_col1.button(toggle_label, key="chat-toggle-compact-tools", use_container_width=True):
        st.session_state["chat_compact_mode"] = not compact_mode
        st.rerun()
    if compact_mode:
        toggle_col2.caption("Clean view enabled: secondary tools are collapsed below.")
    else:
        toggle_col2.caption("Full view enabled: all chat helpers are expanded.")
    selected_hud_mode = toggle_col3.radio(
        "HUD",
        options=["minimal", "balanced", "debug"],
        index=["minimal", "balanced", "debug"].index(hud_mode),
        horizontal=True,
        label_visibility="collapsed",
        key="chat-hud-mode-selector",
    )
    if selected_hud_mode != hud_mode:
        st.session_state["chat_hud_mode"] = selected_hud_mode
        st.rerun()

    delivery = runtime_delivery_snapshot()
    if delivery.get("reconnecting"):
        st.warning("Reconnecting... Dad is retrying with exponential backoff.")
    if int(delivery.get("queued", 0) or 0) > 0:
        queue_col1, queue_col2, queue_col3 = st.columns([2.2, 1.2, 1.2])
        queue_col1.caption(f"Queued messages: {int(delivery.get('queued', 0) or 0)}")
        if queue_col2.button("Retry Queue", key="chat-flush-queue", use_container_width=True):
            sent, remaining = flush_queued_messages(limit=3)
            st.toast(f"Sent {sent} queued message(s). Remaining: {remaining}")
            st.rerun()
        if queue_col3.button("Clear Queue", key="chat-clear-queue", use_container_width=True):
            clear_queued_messages()
            st.rerun()
    if hud_mode in {"balanced", "debug"}:
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        metric_col1.metric("Avg TTFT", f"{int(delivery.get('avg_ttft_ms', 0) or 0)} ms")
        metric_col2.metric("Retry Rate", f"{int(delivery.get('retries', 0) or 0)}")
        metric_col3.metric("Dropout", f"{float(delivery.get('dropout_rate', 0.0) or 0.0) * 100:.1f}%")
        st.caption(f"Reliability state: {reliability_state_from_snapshot(delivery)}")

    control_col1, control_col2, control_col3 = st.columns([1.4, 2.2, 3.4])
    scope = control_col1.radio(
        "Scope",
        options=["thread", "global"],
        horizontal=True,
        key="agent_control_scope",
        label_visibility="collapsed",
    )
    scope_key = "global" if scope == "global" else str(active_thread_id or "default")
    mode_map = cast(dict[str, str], st.session_state.setdefault("agent_mode_by_scope", {}))
    permissions_map = cast(dict[str, list[str]], st.session_state.setdefault("tool_permissions_by_scope", {}))
    current_mode = str(mode_map.get(scope_key) or "balanced")
    mode_choice = control_col2.selectbox(
        "Agent mode",
        options=["balanced", "focus", "tool-heavy"],
        index=["balanced", "focus", "tool-heavy"].index(current_mode) if current_mode in {"balanced", "focus", "tool-heavy"} else 0,
        key="chat-agent-mode",
    )
    mode_map[scope_key] = mode_choice
    st.session_state["agent_mode_by_scope"] = mode_map
    current_permissions = list(permissions_map.get(scope_key) or ["memory_lookup", "photo_generation", "tts"])
    permission_choice = control_col3.multiselect(
        "Tool permissions",
        options=["memory_lookup", "web_lookup", "photo_generation", "tts", "document_read"],
        default=current_permissions,
        key="chat-tool-permissions",
    )
    permissions_map[scope_key] = list(permission_choice)
    st.session_state["tool_permissions_by_scope"] = permissions_map

    with st.expander("Model Compare", expanded=False):
        compare_prompt = st.text_input("Prompt for A/B compare", key="chat-model-compare-prompt")
        cmp_col1, cmp_col2, cmp_col3 = st.columns([2.1, 2.1, 1])
        model_a = cmp_col1.text_input("Model A", value="llama3.2:latest", key="chat-model-a")
        model_b = cmp_col2.text_input("Model B", value="llama3.1:8b", key="chat-model-b")
        if cmp_col3.button("Run", key="chat-run-model-compare", use_container_width=True):
            if compare_prompt.strip():
                left, right = st.columns(2)
                with left:
                    st.caption(f"A: {model_a}")
                    res_a = process_prompt_via_runtime(
                        thread_id=active_thread_id,
                        prompt=compare_prompt,
                        attachments=[],
                        model_override=str(model_a or "").strip(),
                        temperature_style="balanced",
                    )
                    st.markdown(str(res_a.get("reply") or "[no reply]"))
                with right:
                    st.caption(f"B: {model_b}")
                    res_b = process_prompt_via_runtime(
                        thread_id=active_thread_id,
                        prompt=compare_prompt,
                        attachments=[],
                        model_override=str(model_b or "").strip(),
                        temperature_style="balanced",
                    )
                    st.markdown(str(res_b.get("reply") or "[no reply]"))
            else:
                st.info("Enter a compare prompt first.")



    # --- MODERN STREAMLIT CHAT UI ---
    st.markdown("<hr>")
    st.subheader("Dad Chat")

    # Render chat bubbles or placeholder if empty
    if not thread_messages:
        with st.chat_message("assistant", avatar="🧔"):
            st.markdown("_(No messages yet. Say something to start the conversation!)_")
    else:
        for msg in thread_messages:
            role = str(msg.get("role", "user")).lower()
            content = str(msg.get("content", ""))
            if role == "assistant":
                with st.chat_message("assistant", avatar="🧔"):
                    st.markdown(content)
            else:
                with st.chat_message("user", avatar="🧑"):
                    st.markdown(content)

    # Chat input box (always enabled)
    ollama_status = getattr(bot, "model_runtime", None)
    ollama_offline = False
    if ollama_status and hasattr(ollama_status, "ollama_status"):
        status = ollama_status.ollama_status()
        ollama_offline = not bool(status.get("connected", False))
    if ollama_offline:
        st.warning("Ollama is offline. Using fallback model for chat responses.")

    # Always enable chat input
    user_input = st.chat_input("Say something to Dad...", disabled=False, key="chat_input_always_enabled")
    if user_input is not None and user_input.strip() != "":
        st.session_state["nudge_prompt"] = user_input
        # Process the input immediately using process_prompt_via_runtime
        res = process_prompt_via_runtime(
            thread_id=active_thread.get("thread_id", "default"),
            prompt=user_input,
            attachments=[],
            model_override="",  # Let backend pick fallback if Ollama is offline
            temperature_style="balanced",
        )
        reply = res.get("reply") or "[No reply]"
        with st.chat_message("assistant", avatar="🧔"):
            st.markdown(reply)
        st.rerun()

    # --- MINIMAL CHAT UI: Fallback (only shown if chat_input is not available) ---
    # (Kept for legacy/fallback, but will not be shown in modern Streamlit)


def _render_policy_store_editor(bot: "DadBot") -> None:
    store = _resolve_policy_store(bot)
    if store is None:
        st.caption("Policy editor unavailable until policy store is initialized.")
        return

    current_policy, err = _run_sync_policy_call(store.get_current_policy)
    if err:
        st.warning(str(err.get("detail") or "Unable to load policy"))
        return
    if not isinstance(current_policy, DadPolicy):
        st.warning("Policy editor could not load current policy.")
        return

    with st.expander("Policy Editor", expanded=False):
        version = st.text_input("Version", value=str(current_policy.version), key="sidebar-policy-version")
        comment = st.text_input(
            "Comment",
            value=str(current_policy.comment or ""),
            key="sidebar-policy-comment",
        )
        persona_style_raw = st.text_area(
            "Persona Style (JSON)",
            value=json.dumps(dict(current_policy.persona_style), indent=2),
            key="sidebar-policy-persona-style",
            height=120,
        )
        relationship_rules_raw = st.text_area(
            "Relationship Rules (JSON)",
            value=json.dumps(dict(current_policy.relationship_rules), indent=2),
            key="sidebar-policy-relationship-rules",
            height=120,
        )
        safety_boundaries_raw = st.text_area(
            "Safety Boundaries (JSON)",
            value=json.dumps(dict(current_policy.safety_boundaries), indent=2),
            key="sidebar-policy-safety-boundaries",
            height=120,
        )
        memory_preferences_raw = st.text_area(
            "Memory Preferences (JSON)",
            value=json.dumps(dict(current_policy.memory_preferences), indent=2),
            key="sidebar-policy-memory-preferences",
            height=120,
        )

        save_clicked = st.button("Save Policy", use_container_width=True, key="sidebar-policy-save")
        if save_clicked:
            try:
                persona_style = dict(json.loads(persona_style_raw))
                relationship_rules = dict(json.loads(relationship_rules_raw))
                safety_boundaries = dict(json.loads(safety_boundaries_raw))
                memory_preferences = dict(json.loads(memory_preferences_raw))
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                st.error(f"Invalid JSON in editor fields: {exc}")
            else:
                candidate = DadPolicy(
                    version=str(version or "").strip() or str(current_policy.version),
                    persona_style=persona_style,
                    relationship_rules=relationship_rules,
                    safety_boundaries=safety_boundaries,
                    memory_preferences=memory_preferences,
                    created_at=datetime.utcnow().isoformat() + "Z",
                    created_by="streamlit",
                    comment=str(comment or "").strip() or None,
                )
                _, save_err = _run_sync_policy_call(store.save_policy, candidate, comment=candidate.comment)
                if save_err:
                    st.error(str(save_err.get("detail") or "Failed to save policy"))
                else:
                    st.success("Policy saved.")
                    st.rerun()


        history, history_err = _run_sync_policy_call(store.list_history)
        if history_err:
            st.caption(str(history_err.get("detail") or "Policy history unavailable"))
            return
        # If history is a coroutine, run it
        if inspect.iscoroutine(history):
            try:
                history = asyncio.run(history)
            except Exception as exc:
                st.caption(f"Error loading policy history: {exc}")
                return
        history_items = [item for item in list(history or []) if isinstance(item, DadPolicy)]
        versions = [str(item.version) for item in reversed(history_items)]
        if not versions:
            st.caption("No historical policy versions found.")
            return

        rollback_version = st.selectbox(
            "Rollback to version",
            options=versions,
            key="sidebar-policy-rollback-version",
        )
        rollback_comment = st.text_input(
            "Rollback Comment",
            value="",
            key="sidebar-policy-rollback-comment",
        )
        if st.button("Rollback", use_container_width=True, key="sidebar-policy-rollback"):
            _, rollback_err = _run_sync_policy_call(
                store.rollback_to_version,
                rollback_version,
                comment=str(rollback_comment or "").strip() or None,
            )
            if rollback_err:
                st.error(str(rollback_err.get("detail") or "Rollback failed"))
            else:
                st.success(f"Rolled back to {rollback_version}.")
                st.rerun()


def _render_sidebar_onboarding_readiness(bot: "DadBot") -> None:
    api = get_chat_event_api()
    shell = ui_shell_snapshot(bot)
    ollama_status = dict(shell.get("ollama") or {})
    onboarding_complete = bool(st.session_state.get("onboarding_complete") or api.onboarding_complete())
    avatar_ready = bool(api.current_avatar_exists())
    ollama_connected = bool(ollama_status.get("connected"))
    policy_health = _sidebar_policy_store_health(bot)
    policy_ready = str(policy_health.get("status") or "").strip().lower() in {"healthy", "deferred"}

    checks = [
        ("Onboarding complete", onboarding_complete, "Profile seeded" if onboarding_complete else "Setup wizard pending"),
        ("Ollama connection", ollama_connected, str(ollama_status.get("connection_note") or "offline")),
        ("Avatar ready", avatar_ready, "Custom avatar detected" if avatar_ready else "Using emoji fallback"),
        ("Policy store", policy_ready, str(policy_health.get("detail") or "status unavailable")),
    ]
    ready_count = sum(1 for _, ok, _ in checks if ok)

    with st.container(border=True):
        st.markdown("**First-Run Readiness**")
        st.caption(f"{ready_count}/{len(checks)} checks ready")
        for label, ok, detail in checks:
            badge = "✅" if ok else "⚠️"
            st.caption(f"{badge} {label} - {detail}")

        action_col1, action_col2 = st.columns(2)
        if action_col1.button("Re-open Setup", use_container_width=True, key="sidebar-onboarding-reopen"):
            api.set_onboarding_complete(False)
            st.session_state["onboarding_complete"] = False
            st.session_state["onboarding_step"] = 0
            st.rerun()
        if action_col2.button("Run Health Check", use_container_width=True, key="sidebar-onboarding-health"):
            try:
                health = dict(bot.current_runtime_health_snapshot(force=True, log_warnings=False, persist=False) or {})
            except Exception as exc:
                st.warning(f"Health check unavailable: {exc}")
            else:
                level = str(health.get("level") or "green").strip().lower()
                msg = str(health.get("message") or "runtime stable").strip()
                if level == "red":
                    st.error(f"Runtime health: {level.upper()} - {msg}")
                elif level == "yellow":
                    st.warning(f"Runtime health: {level.upper()} - {msg}")
                else:
                    st.success(f"Runtime health: {level.upper()} - {msg}")


@maybe_fragment
def render_sidebar(bot: "DadBot"):
    api = get_chat_event_api()

    # Avatar section with mood-aware styling
    shell = ui_shell_snapshot(bot)
    current_mood = str(shell.get("last_mood") or "neutral").lower()
    mood_display = titleize_token(current_mood)
    mood_colors = {
        "positive": "#3498db",
        "neutral": "#95a5a6",
        "sad": "#9b59b6",
        "frustrated": "#e74c3c",
        "tired": "#f39c12",
    }
    mood_color = mood_colors.get(current_mood, "#95a5a6")

    st.markdown(
        f"<div style='text-align:center; padding:0.5rem; border-radius:12px; "
        f"background:linear-gradient(135deg, rgba({int(mood_color[1:3], 16)}, {int(mood_color[3:5], 16)}, {int(mood_color[5:7], 16)}, 0.1)); "
        f"border:2px solid {mood_color}; animation: pulse 2s infinite;'>"
        f"<style>@keyframes pulse {{ 0%, 100% {{ opacity:1; }} 50% {{ opacity:0.9; }} }}</style>",
        unsafe_allow_html=True,
    )

    avatar_col, gen_col = st.columns([3, 1])
    with avatar_col:
        if DAD_AVATAR_PATH.exists():
            st.image(str(DAD_AVATAR_PATH), width=200)
        else:
            st.markdown(
                f"<div style='font-size:5rem; text-align:center; margin:1rem 0; "
                f"filter: drop-shadow(0 0 8px {mood_color});'>🧔</div>",
                unsafe_allow_html=True,
            )

    with gen_col:
        if st.button("🎨", help="Generate a new AI avatar", key="sidebar-regen-avatar", use_container_width=True):
            with st.spinner("📸"):
                try:
                    ok = api.generate_avatar(mood=current_mood)
                except TypeError:
                    ok = api.generate_avatar()
            if ok:
                st.success("✓")
                st.rerun()
            else:
                DAD_AVATAR_PATH.parent.mkdir(parents=True, exist_ok=True)
                DAD_AVATAR_PATH.write_bytes(_create_placeholder_dad_image())
                st.info("Using local fallback avatar")
                st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)
    st.caption(f"Dad is always here for you, Tony. (Currently {mood_display.lower()})")

    shell = ui_shell_snapshot(bot)
    ollama_status = shell.get("ollama", {})
    connected = bool(ollama_status.get("connected", False))
    model_count = int(ollama_status.get("model_count", 0) or 0)
    connection_note = str(ollama_status.get("connection_note") or "offline")
    indicator_color = "#1fa75a" if connected else "#c0392b"
    indicator_text = "Ollama Online" if connected else "Ollama Offline"
    st.markdown(
        f"<div style='display:flex;align-items:center;gap:0.55rem;margin:0.3rem 0 0.7rem;'>"
        f"<span style='width:12px;height:12px;border-radius:999px;background:{indicator_color};display:inline-block;'></span>"
        f"<strong>{indicator_text}</strong></div>",
        unsafe_allow_html=True,
    )
    st.caption(f"{connection_note}. Models detected: {model_count}")

    with st.container(border=True):
        st.markdown("**Dad Pulse**")
        pulse_col1, pulse_col2 = st.columns(2)
        pulse_col1.metric("Reputation", f"{int(shell.get('reputation_score', 0) or 0)}/100")
        pulse_col2.metric("Mood", titleize_token(str(shell.get("last_mood") or "neutral")))
        circuit_breaker = dict(shell.get("circuit_breaker") or {})
        confidence = float(circuit_breaker.get("reasoning_confidence", 1.0) or 1.0)
        st.caption(f"Reasoning confidence: {confidence:.0%}")
        if circuit_breaker.get("active"):
            st.warning(str(circuit_breaker.get("message") or "Dad needs a quick clarification."))
            for index, prompt in enumerate(circuit_breaker.get("suggested_prompts", [])[:3], start=1):
                if st.button(prompt, key=f"sidebar-clarify-{index}", use_container_width=True):
                    st.session_state.nudge_prompt = prompt
                    st.session_state.primary_view = "chat"
                    st.rerun()
        else:
            st.caption("Dad is tracking your thread cleanly right now.")

    local_mcp = dict(shell.get("local_mcp") or {})
    with st.container(border=True):
        st.markdown("**Local MCP**")
        if local_mcp.get("available"):
            state_entries = int(local_mcp.get("local_state_entries", 0) or 0)
            entry_label = "entry" if state_entries == 1 else "entries"
            st.caption(
                f"dadbot-local-services is ready with {int(local_mcp.get('tool_count', 0) or 0)} tools and {state_entries} local state {entry_label}."
            )
        else:
            st.caption(
                "Install the optional MCP dependency to expose Dad's local reminder, calendar, email, and state tools."
            )
        st.caption(f"Narrative memories distilled: {int(shell.get('narrative_memory_count', 0) or 0)}")

    with st.container(border=True):
        st.markdown("**Workshop Controls**")
        st.caption("Health, diagnostics, and tuning live in Dad's Workshop.")
        if st.button("Open Dad's Workshop", use_container_width=True, key="sidebar-open-workshop"):
            st.session_state.primary_view = "workshop"
            st.session_state["workshop-section"] = "Status"
            st.rerun()

    with st.container(border=True):
        st.markdown("**Policy Store**")
        policy_health = _sidebar_policy_store_health(bot)
        st.caption(f"Current policy version: {policy_health.get('version', 'unknown')}")
        status = str(policy_health.get("status") or "unknown").lower()
        detail = str(policy_health.get("detail") or "")
        if status == "healthy":
            st.success(detail or "policy store healthy")
        elif status == "deferred":
            st.info(detail or "policy check deferred")
        else:
            st.warning(detail or "policy store health check unavailable")
        _render_policy_store_editor(bot)

    with st.container(border=True):
        st.markdown("**Execution Path**")
        st.caption("Canonical thin-spine runtime path is active.")

    _render_sidebar_onboarding_readiness(bot)

    st.subheader("Threads")
    all_threads = list(api.list_chat_threads() or [])
    open_only = st.checkbox("Show open threads only", value=True, key="sidebar-thread-open-only")
    search_query = str(st.text_input("Search threads", value="", key="sidebar-thread-search") or "").strip().lower()
    sort_mode = st.selectbox(
        "Sort",
        options=["Pinned + recent", "Recent"],
        index=0,
        key="sidebar-thread-sort",
    )
    visible_threads = [thread for thread in all_threads if (not open_only or not thread.get("closed"))]
    if search_query:
        visible_threads = [
            thread
            for thread in visible_threads
            if search_query in str(thread.get("title") or "").lower()
            or search_query in str(thread.get("last_message") or "").lower()
        ]
    meta_map = thread_ui_metadata()
    if sort_mode == "Pinned + recent":
        visible_threads = sorted(
            visible_threads,
            key=lambda thread: (
                0 if bool(dict(meta_map.get(str(thread.get("thread_id") or "")) or {}).get("pinned", False)) else 1,
                str(thread.get("updated_at") or ""),
            ),
            reverse=False,
        )
        visible_threads = list(reversed(visible_threads))
    else:
        visible_threads = sorted(visible_threads, key=lambda thread: str(thread.get("updated_at") or ""), reverse=True)
    visible_threads = visible_threads or all_threads
    thread_labels = []
    label_to_thread_id = {}
    active_label = None
    for thread in visible_threads:
        thread_id = str(thread.get("thread_id") or "")
        label = thread_label_with_meta(thread)
        thread_labels.append(label)
        label_to_thread_id[label] = thread_id
        if thread_id == api.active_thread_id:
            active_label = label
    if thread_labels:
        selected_label = st.selectbox(
            "Switch thread",
            options=thread_labels,
            index=thread_labels.index(active_label) if active_label in thread_labels else 0,
            key="sidebar-thread-picker",
        )
        selected_thread_id = label_to_thread_id.get(selected_label)
        if st.button(
            "Go to thread",
            use_container_width=True,
            disabled=not selected_thread_id or selected_thread_id == api.active_thread_id,
        ):
            switch_active_thread(str(selected_thread_id))
            st.rerun()
        selected_thread = next(
            (t for t in visible_threads if str(t.get("thread_id") or "") == str(selected_thread_id or "")), {}
        )
        st.caption(str(selected_thread.get("last_message") or "Fresh chat"))

        selected_meta = dict(meta_map.get(str(selected_thread_id or "")) or {})
        selected_title = str(selected_meta.get("title") or selected_thread.get("title") or "Chat")
        title_edit = st.text_input("Rename thread", value=selected_title, key="sidebar-thread-rename")
        rename_col1, rename_col2 = st.columns(2)
        if rename_col1.button("Save Name", use_container_width=True, key="sidebar-thread-rename-save"):
            selected_meta["title"] = str(title_edit or "").strip() or selected_title
            meta_map[str(selected_thread_id or "")] = selected_meta
            st.session_state["thread_ui_meta"] = meta_map
            st.rerun()
        pin_label = "Unpin" if bool(selected_meta.get("pinned", False)) else "Pin"
        if rename_col2.button(pin_label, use_container_width=True, key="sidebar-thread-pin-toggle"):
            selected_meta["pinned"] = not bool(selected_meta.get("pinned", False))
            meta_map[str(selected_thread_id or "")] = selected_meta
            st.session_state["thread_ui_meta"] = meta_map
            st.rerun()

        arch_col1, arch_col2 = st.columns(2)
        if arch_col1.button("Archive", use_container_width=True, key="sidebar-thread-archive"):
            if selected_thread_id:
                api.mark_chat_thread_closed(thread_id=str(selected_thread_id), closed=True)
                st.rerun()
        if arch_col2.button("Unarchive", use_container_width=True, key="sidebar-thread-unarchive"):
            if selected_thread_id:
                api.mark_chat_thread_closed(thread_id=str(selected_thread_id), closed=False)
                st.rerun()
    st.subheader("Quick actions")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("New Thread", use_container_width=True, type="primary"):
            create_new_thread()
            st.rerun()
    with col2:
        if st.button("📸 Send Photo", use_container_width=True):
            if emit_generated_photo_message(bot, str(api.active_thread_id or "default")):
                st.rerun()
    col3, col4 = st.columns(2)
    with col3:
        if st.button("Daily Check-in", use_container_width=True):
            st.session_state.nudge_prompt = "Hey Dad, just checking in today. How are things on your end?"
            st.rerun()
    with col4:
        if st.button("Evolve Persona", use_container_width=True):
            with st.spinner("Running memory synthesis and persona evolution..."):
                try:
                    get_chat_event_api().consolidate_memories(force=True)
                    st.toast("Persona evolution complete!", icon="🧔")
                except Exception as exc:
                    st.error(f"Could not evolve persona: {exc}")
    _sidebar_thread_id = api.active_thread_id
    _sidebar_thread_id = str(_sidebar_thread_id or "default")
    _sidebar_msgs = list(
        load_thread_projection(
            api=api,
            thread_id=_sidebar_thread_id,
            seed_messages=bot_messages_for_thread(_sidebar_thread_id),
        ).messages
        or []
    )
    _sidebar_chat_export = json.dumps(
        [{"role": m.get("role"), "content": m.get("content")} for m in _sidebar_msgs],
        indent=2,
    ).encode("utf-8")
    st.download_button(
        "Export Chat",
        data=_sidebar_chat_export,
        file_name=f"dadbot_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
        use_container_width=True,
    )
    st.divider()
    st.caption("Built with love for Tony")


def render_mobile_bottom_navigation():
    st.markdown("<div class='dad-mobile-bottom-nav'>", unsafe_allow_html=True)
    active_view = state_manager.active_primary_view()
    nav_cols = st.columns(4)
    if nav_cols[0].button(
        "💬 Chat",
        key="bottom-nav-chat",
        use_container_width=True,
        type="primary" if active_view == "chat" else "secondary",
    ):
        state_manager.set_primary_view("chat")
        st.rerun()
    if nav_cols[1].button(
        "🩺 Status",
        key="bottom-nav-status",
        use_container_width=True,
        type="primary" if active_view == "status" else "secondary",
    ):
        state_manager.set_primary_view("status")
        st.rerun()
    if nav_cols[2].button(
        "🛠️ Workshop",
        key="bottom-nav-workshop",
        use_container_width=True,
        type="primary" if active_view == "workshop" else "secondary",
    ):
        state_manager.set_primary_view("workshop")
        st.rerun()
    if nav_cols[3].button(
        "🎙️ Voice",
        key="bottom-nav-voice",
        use_container_width=True,
        type="primary" if active_view == "voice" else "secondary",
    ):
        state_manager.set_primary_view("voice")
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)


def active_screen_mode() -> str:
    state_mode = str(st.session_state.get("screen_mode") or "").strip().lower()
    if state_mode in {"chat", "app"}:
        return state_mode
    try:
        raw = str(st.query_params.get("screen") or "").strip().lower()
    except Exception:
        raw = ""
    return "chat" if raw == "chat" else "app"


def set_screen_mode(mode: str) -> None:
    target = str(mode or "app").strip().lower()
    st.session_state["screen_mode"] = "chat" if target == "chat" else "app"
    try:
        if target == "chat":
            st.query_params["screen"] = "chat"
        elif "screen" in st.query_params:
            del st.query_params["screen"]
    except Exception:
        # Session state routing remains authoritative even if URL sync fails.
        return


@maybe_fragment
def render_workshop_tab(bot: "DadBot"):
    st.markdown(
        "<div class='dad-workshop-shell'><strong>Dad's Workshop</strong><br><span style='opacity:0.8;'>Tune Dad, check status, and browse your notebook.</span></div>",
        unsafe_allow_html=True,
    )
    section = st.radio(
        "Workshop section",
        options=["Status", "Preferences", "Data", "Mobile"],
        horizontal=True,
        label_visibility="collapsed",
        key="workshop-section",
    )
    api = get_chat_event_api()
    if section == "Status":
        render_status_tab(bot)
    elif section == "Preferences":
        render_preferences_tab(cast("DadBot", api))
    elif section == "Data":
        render_data_tab(cast("DadBot", api))
    else:
        render_mobile_tab(bot)


def main():
    import dadbot.core.boot_mixin
    import sys
    print("[DEBUG] boot_mixin.py loaded from:", dadbot.core.boot_mixin.__file__, file=sys.stderr)

    st.set_page_config(page_title="Dad Bot", page_icon="🧔", layout="centered", initial_sidebar_state="expanded")

    # Debug bypass: skip wizard
    if st.query_params.get("skip_wizard") == "true":
        st.session_state["onboarding_complete"] = True
        api = get_chat_event_api()
        if hasattr(api, "set_onboarding_complete"):
            api.set_onboarding_complete(True)
        if hasattr(api, "save_profile"):
            api.save_profile()
        st.session_state["primary-nav"] = "chat"
        st.rerun()

    bot = REAL_DADBOT
    if not isinstance(bot.thread_snapshots, dict):
        bot.thread_snapshots = {}

    initialize_session(bot)
    apply_ui_preferences(bot)
    update_ui_mood(bot)
    inject_custom_css(ui_preferences())
    inject_pwa_metadata(ui_preferences())


    # --- DEBUG PANEL (removed for clean UI; gate with debug flag if needed) ---


    require_pin(bot)
    render_runtime_rejection_banner(dismiss_key="dismiss-runtime-rejection-main")


    # Run wizard only if not bypassed
    if not st.session_state.get("onboarding_complete", False):
        try:
            # first_run_wizard(bot)  # Temporarily disabled to avoid onboarding freeze
            pass
        except Exception as e:
            st.error(f"Wizard error: {e}")
            return
    else:
        # Ensure profile is marked complete
        api = get_chat_event_api()
        if hasattr(api, "set_onboarding_complete"):
            api.set_onboarding_complete(True)
        if hasattr(api, "save_profile"):
            api.save_profile()
        if st.session_state.get("primary-nav") != "chat":
            st.session_state["primary-nav"] = "chat"
            st.rerun()


    active_thread = get_chat_event_api().active_chat_thread() or create_new_thread()

    # --- Navigation ---

    nav_options = ["status", "chat", "workshop", "voice"]
    current_nav = st.session_state.get("primary-nav") or state_manager.active_primary_view()
    if current_nav not in nav_options:
        current_nav = "chat"
    nav_choice = st.radio(
        "Navigate",
        options=nav_options,
        index=nav_options.index(current_nav),
        horizontal=True,
        format_func=lambda value: {
            "chat": "💬 Chat",
            "status": "🩺 Status",
            "workshop": "🛠️ Dad's Workshop",
            "voice": "🎙️ Voice",
        }[value],
        key="primary-nav",
    )
    state_manager.set_primary_view(nav_choice)
    with st.sidebar:
        render_sidebar(bot)


    # --- Tab rendering with appropriate bot instance ---
    if nav_choice == "status":
        render_status_tab(bot)
    elif nav_choice == "chat":
        try:
            render_chat_tab(bot, active_thread)
        except Exception as exc:
            import traceback
            st.error(f"Chat tab failed: {exc}\n" + traceback.format_exc())
    elif nav_choice == "workshop":
        render_workshop_tab(bot)
    else:
        with st.container(border=True):
            st.subheader("Talk to Dad")
            st.caption("Voice-first mode for quick hands-free check-ins.")
            _voice_prompt = render_voice_controls(bot)
            if _voice_prompt:
                st.info("Voice captured. Switch to Chat to send or edit this message.")
        with st.expander("Live Voice Call (Experimental)", expanded=True):
            render_realtime_voice_call(bot)
            voice_mode = str(voice_preferences().get("mode") or "push_to_talk")
            if voice_mode == "ambient":
                with st.container(border=True):
                    st.subheader("🌐 Ambient Listener")
                    st.caption("Dad is always listening. Speak naturally — no button needed.")
                    render_ambient_voice_listener(bot)
                    pending = list(st.session_state.get("ambient_utterance_queue") or [])
                    if pending:
                        st.success(f"{len(pending)} utterance(s) queued — switch to Chat to review or auto-send.")


    render_mobile_bottom_navigation()

# --- Ensure Streamlit always runs main() ---
main()
