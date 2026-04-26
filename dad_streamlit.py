from __future__ import annotations

import base64
import hashlib
import importlib
import json
import logging
import mimetypes
import os
import re
import shutil
import subprocess
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlencode
from urllib.request import urlopen
from urllib.error import URLError

import numpy as np
import ollama
import streamlit as st
import streamlit.components.v1 as components

from dadbot.heritage_import import build_heritage_memories
from dadbot.runtime_core.streamlit_runtime import StreamlitRuntime
from dadbot.ui.utils import (
    maybe_fragment,
    filter_memory_entries,
    option_index,
    titleize_token,
    enabled_label,
)
from dadbot.ui.utils import ambient_fragment
from dadbot.ui.prefs_state import (
    default_ui_preferences,
    _voice_defaults,
    ui_preferences,
    voice_preferences,
    profile_voice_preferences,
    sync_ui_voice_from_profile,
    notification_settings,
    voice_profile_catalog,
)
from dadbot.ui.voice_control_plane import VoiceSessionController
from dadbot.ui.helpers import (
    apply_power_mode,
    apply_ui_preferences,
    find_available_image_model,
    fetch_ical_events,
    export_bundle_payload,
    render_voice_dependency_help,
    local_stt_backend_status,
    local_tts_backend_status,
    heritage_files_with_limits,
)
from dadbot.ui.preferences import render_preferences_tab
from dadbot.ui.data import render_memory_garden, render_data_tab
from dadbot.ui import interaction_controller, state_manager
from dadbot.consumers.streamlit import load_thread_projection
from dadbot.runtime_core import UIRuntimeAPI, ThreadView

if TYPE_CHECKING:
    from dadbot.core.dadbot import DadBot

try:
    STAGE_METADATA = getattr(importlib.import_module("dadbot.core.system_behavior_views"), "STAGE_METADATA")
except Exception:  # pragma: no cover - optional compatibility shim
    STAGE_METADATA = {}

logger = logging.getLogger(__name__)

RUNTIME_REJECTION_SESSION_KEY = state_manager.RUNTIME_REJECTION_SESSION_KEY
RUNTIME_TURN_TIMELINE_SESSION_KEY = state_manager.RUNTIME_TURN_TIMELINE_SESSION_KEY
EVENT_TO_UI_SIGNAL = state_manager.EVENT_TO_UI_SIGNAL
SIGNAL_PRIORITY = state_manager.SIGNAL_PRIORITY

try:
    from faster_whisper import WhisperModel
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
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration as WebRtcRTCConfiguration
    _WEBRTC_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    _WEBRTC_AVAILABLE = False

STATIC_DIR = Path("static")
DAD_AVATAR_PATH = STATIC_DIR / "dad_avatar.png"
STATIC_DIR.mkdir(exist_ok=True)

INITIAL_GREETING = "That's my boy. I love hearing that, Tony."
DEFAULT_EXPORT_PATH = Path("memory_export.json")
PWA_MANIFEST_FILE = "dadbot-manifest.webmanifest"
PWA_ICON_FILE = "dadbot-icon.svg"
PWA_MASKABLE_ICON_FILE = "dadbot-icon-maskable.svg"
PWA_HEAD_SYNC_FILE = "dadbot-head-sync.html"
UI_STYLESHEET_FILE = "dadbot-ui.css"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
CHAT_EVENT_JOURNAL_PATH = Path("runtime") / "chat_runtime_events.jsonl"


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


def record_turn_timeline_event(*, thread_id: str, event_type: str, summary: str, payload: dict | None = None, severity: str = "info") -> None:
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
                "summary": ", ".join(
                    key for key, value in dict(decision.get("decisions") or {}).items() if bool(value)
                ) or "no side effects",
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


def record_turn_inspector_from_runtime_result(*, thread_id: str, prompt: str, runtime_result: dict, view: ThreadView) -> None:
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
        record_turn_timeline_event(thread_id=normalized_thread, event_type="photo_request", summary="photo effect requested")
    if _event_presence(events_by_type.get("tts_request")):
        record_turn_timeline_event(thread_id=normalized_thread, event_type="tts_request", summary="tts effect requested")

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
    timeline = [
        item
        for item in runtime_turn_timeline()
        if str(item.get("thread_id") or "") == normalized_thread
    ]
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
                    guard_text = ", ".join(
                        f"{item.get('name')}={'ok' if item.get('passed') else 'blocked'}"
                        for item in guard_items
                    ) or "none"
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


def ui_shell_snapshot(bot):
    _ = bot
    return get_chat_event_api().ui_shell_snapshot()


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


def process_prompt_via_runtime(*, thread_id: str, prompt: str, attachments: list[dict] | None = None) -> dict:
    return interaction_controller.process_prompt_via_runtime(thread_id=thread_id, prompt=prompt, attachments=attachments)


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
                    "note": f"Tony uploaded {str(getattr(uploaded_file, 'name', 'an image'))}",
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
        index=known_devices.index(str(voice.get("last_used_device") or "default")) if str(voice.get("last_used_device") or "default") in known_devices else 0,
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
        audio_label = "Hold mic, speak, release" if str(voice.get("mode") or "push_to_talk") == "push_to_talk" else "Always-listening capture"
        st.caption("WebRTC unavailable; using Streamlit audio input fallback.")
        clip = st.audio_input(audio_label, key=f"{key_prefix}-audio-fallback")
        if clip is None:
            return b""
        return bytes(clip.getvalue() or b"")

    media_audio: dict | bool = True
    if str(selected_device or "default") != "default":
        media_audio = {"deviceId": {"exact": str(selected_device)}}

    rtc_config = WebRtcRTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
    webrtc_ctx = webrtc_streamer(
        key=f"{key_prefix}-webrtc-capture",
        mode=WebRtcMode.SENDONLY,
        rtc_configuration=rtc_config,
        media_stream_constraints={"video": False, "audio": media_audio},
        async_processing=True,
    )

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

        default_rate = int(engine.getProperty("rate") or 180)
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
def render_voice_controls(bot: DadBot):
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
        "<span style='width:10px;height:10px;border-radius:50%;background:#22c55e;"
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
    st.toast(f"Dad heard: \"{transcript_text[:60]}\"", icon="🎙️")
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
        st.info(
            "Install `streamlit-webrtc` to enable real-time voice calls:\n"
            "```\npip install streamlit-webrtc\n```"
        )
        return

    if not voice.get("enabled", False):
        st.info("Enable voice in Preferences → Voice to use the real-time call feature.")
        return

    st.subheader("📞 Talk to Dad — Live")
    st.caption("Hands-free, real-time voice conversation. Uses your mic → STT → Dad → TTS pipeline.")

    known_devices = _voice_known_devices(voice, controller.runtime_state if isinstance(controller.runtime_state, dict) else {})
    selected_device = st.selectbox(
        "Realtime call input device ID",
        options=known_devices,
        index=known_devices.index(str(voice.get("last_used_device") or "default")) if str(voice.get("last_used_device") or "default") in known_devices else 0,
        key="webrtc-call-device",
    )
    controller.set_device(selected_device)
    _persist_known_devices(voice, known_devices)

    rtc_config = WebRtcRTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    media_audio: dict | bool = True
    if str(selected_device or "default") != "default":
        media_audio = {"deviceId": {"exact": str(selected_device)}}

    webrtc_ctx = webrtc_streamer(
        key="dadbot-voice-call",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_config,
        media_stream_constraints={"video": False, "audio": media_audio},
        async_processing=True,
    )

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


def render_reply_tts(bot: DadBot, reply_text):
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
        (
            text
            + "|"
            + str(voice.get("tts_voice") or "warm_dad")
            + "|"
            + str(int(voice.get("tts_rate") or 0))
        ).encode("utf-8")
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
            --dad-primary: {palette['primary']};
            --dad-accent: {palette['accent']};
            --dad-bg: {palette['bg']};
            --dad-surface: {palette['surface']};
            --dad-surface-alt: {palette['surface_alt']};
            --dad-text: {palette['text']};
            --dad-muted: {palette['muted']};
            --dad-border: {palette['border']};
            --dad-hero-start: {palette['hero_start']};
            --dad-hero-end: {palette['hero_end']};
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


def render_hero(bot: DadBot, active_thread: dict):
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
                Thread: <strong>{active_thread.get('title', 'General Chat')}</strong>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_status_strip(bot: DadBot):
    api = get_chat_event_api()
    living = api.living_dad_snapshot(limit=2)
    cards = [
        (str(len(api.list_chat_threads())), "Threads"),
        (str(len(api.reminder_catalog())), "Open reminders"),
        (str(living["counts"]["proactive_queue"]), "Queued check-ins"),
    ]
    card_markup = "".join(
        f'<div class="status-card"><strong>{value}</strong><span>{label}</span></div>'
        for value, label in cards
    )
    st.markdown(f'<div class="status-strip">{card_markup}</div>', unsafe_allow_html=True)


def render_mobile_tab(bot: DadBot):
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

        selected_thread_label = st.selectbox(
            "Jump to thread without opening the sidebar",
            options=list(thread_labels.keys()),
            index=list(thread_labels.keys()).index(current_selection) if current_selection in thread_labels else 0,
        ) if thread_labels else None
        selected_thread_id = thread_labels.get(selected_thread_label) if selected_thread_label else None
        if st.button("Switch to Selected Thread", use_container_width=True, disabled=not selected_thread_id or selected_thread_id == api.active_thread_id):
            switch_active_thread(selected_thread_id)
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


def update_ui_mood(bot: DadBot):
    _ = bot
    state_manager.set_ui_mood(str(get_chat_event_api().last_saved_mood() or "neutral"))


def optimize_runtime_for_hardware(bot: DadBot):
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
    apply_ui_preferences(api)
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


def initialize_session(bot: DadBot):
    api = get_chat_event_api()
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


def first_run_wizard(bot: DadBot):
    """One-time onboarding wizard – runs only on the very first launch.

    Detected via 'onboarding_complete' in session_state. Once the user
    clicks Finish, the flag is persisted to the profile so it never shows again.
    """
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
            st.info("Piper requires the `piper` binary and an `.onnx` model file. See Preferences → Voice to configure after setup.")
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
                st.warning("Could not generate avatar right now – using emoji fallback. You can try again from Preferences.")
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

    st.stop()


def require_pin(bot: DadBot):
    api = get_chat_event_api()
    security = api.streamlit_security_settings()
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
    st.stop()


def generate_dad_photo():
    with st.spinner("Dad is taking a quick photo for you..."):
        try:
            candidates = ["flux", "flux-dev", "flux-schnell", "sdxl", "stable-diffusion"]
            model = find_available_image_model(tuple(candidates))
            if not model:
                st.warning("No image model found. Pull 'flux' or 'sdxl' in Ollama for best results.")
                return None
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


def emit_generated_photo_message(bot: DadBot, thread_id: str, message: str = "Here's a quick photo I took for you, buddy. Love you.") -> bool:
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


def render_companion_toolbar(bot: DadBot):
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


def render_chat_tab(bot: DadBot, active_thread: dict):
    initialize_session(bot)
    api = get_chat_event_api()
    render_runtime_guardrails_card(dismiss_key="dismiss-runtime-guardrails-chat")
    render_runtime_semantic_strip(thread_id=str(api.active_thread_id or "default"))
    active_thread_id = str(api.active_thread_id or "default")
    thread_messages = list(
        load_thread_projection(
            api=api,
            thread_id=active_thread_id,
            seed_messages=bot_messages_for_thread(active_thread_id),
        ).messages
        or []
    )
    render_hero(bot, active_thread)
    render_status_strip(bot)
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
    dashboard = api.dashboard_status_snapshot()
    graph_fallback = dict(dashboard.get("graph_fallback") or {})
    if graph_fallback.get("active"):
        mode = str(graph_fallback.get("degraded_mode") or "legacy").replace("_", " ")
        count = int(graph_fallback.get("event_count", 0) or 0)
        st.warning(
            f"Turn graph degraded to {mode}. Dad is continuing on legacy turn processing ({count} recent event{'s' if count != 1 else ''})."
        )

    queued_nudges = list(api.pending_proactive_messages())
    if queued_nudges:
        with st.container(border=True):
            st.subheader("Proactive nudges")
            for index, entry in enumerate(queued_nudges[:4], start=1):
                source = str(entry.get("source") or "general").replace("-", " ").title()
                message = str(entry.get("message") or "").strip()
                if not message:
                    continue
                cols = st.columns([5, 1])
                cols[0].markdown(f"**{source}:** {message}")
                if cols[1].button("Discuss", key=f"nudge-discuss-{index}", use_container_width=True):
                    st.session_state.nudge_prompt = message

    pending_attachments: list = []
    attachment_issues: list = []
    voice_prompt = None
    with st.container(border=True):
        st.subheader("Message Tools")
        st.caption("Keep this simple: attach when needed, or tap the mic and talk to Dad.")
        tools_col1, tools_col2 = st.columns(2)
        with tools_col1:
            with st.expander("📎 Attach files or photos", expanded=False):
                st.caption("Drag-and-drop images or documents to give Dad richer context.")
                dropped_files = st.file_uploader(
                    "Drop images or documents",
                    type=["png", "jpg", "jpeg", "webp", "txt", "md", "csv", "json", "pdf", "log"],
                    accept_multiple_files=True,
                    key="chat-dropzone-files",
                )
                pending_attachments, attachment_issues = build_chat_attachments_from_uploads(dropped_files)
                if attachment_issues:
                    for issue in attachment_issues[:4]:
                        st.warning(issue)
                if pending_attachments:
                    for attachment in pending_attachments:
                        attachment_type = str(attachment.get("type") or "").strip().lower()
                        if attachment_type == "image":
                            image_name = str(attachment.get("name") or "image")
                            st.caption(f"Image ready: {image_name}")
                            try:
                                st.image(
                                    base64.b64decode(attachment.get("image_b64", "")),
                                    caption=image_name,
                                    width=220,
                                )
                            except Exception:
                                st.caption("[preview unavailable]")
                        elif attachment_type == "document":
                            document_name = str(attachment.get("name") or "document")
                            st.caption(f"Document ready: {document_name}")
                            excerpt = str(attachment.get("text") or "").strip()
                            if excerpt:
                                with st.expander(f"Preview: {document_name}", expanded=False):
                                    st.write(excerpt)
        with tools_col2:
            with st.expander("🎙️ Talk to Dad", expanded=bool(st.session_state.get("voice_panel_open", False))):
                voice_prompt = render_voice_controls(bot)
            st.session_state["voice_panel_open"] = False

    nudge_prompt = str(st.session_state.pop("nudge_prompt", "") or "").strip()
    if len(thread_messages) <= 1 and not nudge_prompt:
        st.caption("Try a starter prompt:")
        starter_col1, starter_col2, starter_col3 = st.columns(3)
        if starter_col1.button("How am I doing lately?", key="starter-progress", use_container_width=True):
            nudge_prompt = "Hey Dad, based on our recent talks, how am I doing lately?"
        if starter_col2.button("I need perspective", key="starter-perspective", use_container_width=True):
            nudge_prompt = "Hey Dad, I need perspective on what I'm dealing with right now."
        if starter_col3.button("Quick encouragement", key="starter-encouragement", use_container_width=True):
            nudge_prompt = "Hey Dad, can you give me a short pep talk?"
    last_assistant_idx = max(
        (i for i, msg in enumerate(thread_messages) if msg.get("role") == "assistant"),
        default=-1,
    )
    for _msg_idx, msg in enumerate(thread_messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            for att in msg.get("attachments", []):
                if att.get("type") == "image":
                    try:
                        data = base64.b64decode(att.get("data_b64", ""))
                        st.image(data, caption=att.get("note") or "Photo from Dad", use_container_width=True)
                    except Exception:
                        st.caption("[Photo could not be displayed]")
                if att.get("type") == "document":
                    doc_name = str(att.get("name") or "document")
                    doc_note = str(att.get("note") or "")
                    st.caption(f"Document: {doc_name}")
                    if doc_note:
                        st.caption(doc_note)
            if msg.get("role") == "assistant" and _msg_idx == last_assistant_idx and _msg_idx > 0:
                react_cols = st.columns([1.1, 1.1, 1.1, 6])
                with react_cols[0]:
                    if st.button("❤️ Helpful", key=f"react-helpful-{_msg_idx}", help="This reply felt supportive"):
                        get_chat_event_api().apply_relationship_feedback("supportive")
                        st.toast("Glad that helped, buddy.", icon="❤️")
                with react_cols[1]:
                    if st.button("😌 A bit much", key=f"react-harsh-{_msg_idx}", help="This felt a little harsh"):
                        get_chat_event_api().apply_relationship_feedback("distant")
                        st.toast("Got it — I'll soften my tone next time.", icon="💙")
                with react_cols[2]:
                    if st.button("🌟 More warmth", key=f"react-warm-{_msg_idx}", help="I'd like even more warmth"):
                        get_chat_event_api().apply_relationship_feedback("supportive")
                        st.toast("Noted. Warming it up.", icon="🌟")
    text_prompt = st.chat_input("Talk to Dad...")
    prompt = ""
    if voice_prompt:
        prompt = str(voice_prompt).strip()
    elif text_prompt:
        prompt = str(text_prompt).strip()
    elif nudge_prompt:
        prompt = str(nudge_prompt).strip()
        # Drain ambient voice queue (populated by render_ambient_voice_listener)
        ambient_queue = list(st.session_state.get("ambient_utterance_queue") or [])
        if not prompt and ambient_queue:
            prompt = str(ambient_queue.pop(0)).strip()
            st.session_state["ambient_utterance_queue"] = ambient_queue
    if prompt:
        active_thread = api.active_chat_thread() or active_thread
        if active_thread.get("closed"):
            st.warning("This chat has ended. Start a new thread.")
            st.stop()
        with st.chat_message("user"):
            st.markdown(prompt)
            for attachment in pending_attachments:
                if attachment.get("type") == "image":
                    try:
                        st.image(base64.b64decode(attachment.get("image_b64", "")), caption=attachment.get("name") or "Uploaded image", use_container_width=True)
                    except Exception:
                        st.caption("[Uploaded image unavailable]")
                if attachment.get("type") == "document":
                    st.caption(f"Document: {attachment.get('name', 'document')}")
        with st.chat_message("assistant"):
            with st.spinner("Dad is thinking..."):
                try:
                    runtime_result = process_prompt_via_runtime(
                        thread_id=str(active_thread.get("thread_id") or api.active_thread_id or "default"),
                        prompt=prompt,
                        attachments=pending_attachments,
                    )
                    reply = str(runtime_result.get("reply") or "")
                    should_end = bool(runtime_result.get("should_end", False))
                    assistant_attachments = []
                    mood = str(runtime_result.get("mood") or "neutral")
                    thread_id = str(active_thread.get("thread_id") or api.active_thread_id or "default")
                    if bool(runtime_result.get("photo_requested", False)):
                        photo = generate_dad_photo()
                        if photo:
                            attachment = {
                                "type": "image",
                                "data_b64": base64.b64encode(photo).decode(),
                                "note": "Dad took a quick photo for you",
                            }
                            api.emit_assistant_attachment(
                                thread_id=thread_id,
                                attachment=attachment,
                            )
                            api.process_until_idle(max_events=8)
                            assistant_attachments.append(attachment)
                    st.markdown(reply)
                    render_agentic_trace(runtime_result)
                    voice = voice_preferences()
                    if bool(voice.get("enabled")) and bool(voice.get("tts_enabled", True)):
                        render_reply_tts(bot, reply)
                    latest_view = load_thread_projection(api=api, thread_id=thread_id)
                    record_turn_inspector_from_runtime_result(
                        thread_id=thread_id,
                        prompt=prompt,
                        runtime_result=runtime_result,
                        view=latest_view,
                    )
                    latest_messages = list(latest_view.messages or [])
                    latest_assistant = latest_messages[-1] if latest_messages else {}
                    for att in list(latest_assistant.get("attachments") or assistant_attachments):
                        if att.get("type") == "image":
                            st.image(base64.b64decode(att.get("data_b64", "")), caption=att.get("note") or "Dad took a quick photo for you", use_container_width=True)
                    if should_end:
                        api.mark_chat_thread_closed(closed=True)
                    api.sync_active_thread_snapshot()
                    dashboard_after_turn = api.dashboard_status_snapshot()
                    graph_fallback_after_turn = dict(dashboard_after_turn.get("graph_fallback") or {})
                    if graph_fallback_after_turn.get("active"):
                        mode = str(graph_fallback_after_turn.get("degraded_mode") or "legacy").replace("_", " ")
                        message = str(graph_fallback_after_turn.get("message") or "").strip()
                        st.warning(message or f"Graph degraded and switched to {mode}. Dad continued with legacy turn processing.")
                except Exception as exc:
                    record_runtime_rejection(exc, action="chat_turn")
                    record_turn_timeline_event(
                        thread_id=str(active_thread.get("thread_id") or api.active_thread_id or "default"),
                        event_type="guardrail_rejection",
                        summary=str(exc),
                        payload=_runtime_rejection_payload(exc, action="chat_turn"),
                        severity="warning",
                    )
                    reply = "I'm having trouble saving my thoughts right now - let me try again in a moment."
                    st.error(f"Dad Runtime blocked or failed this request: {exc}")
                    st.markdown(reply)
        api.persist_conversation_async()
        st.rerun()

    active_thread_id = str(api.active_thread_id or "default")
    view = load_thread_projection(api=api, thread_id=active_thread_id)
    render_turn_inspector(thread_id=active_thread_id, view=view)


@maybe_fragment
def render_status_tab(bot: DadBot):
    """Clean, calm, emotionally warm status dashboard."""
    api = get_chat_event_api()
    dashboard = api.dashboard_status_snapshot()
    shell = ui_shell_snapshot(bot)
    health = dashboard.get("health", {})
    relationship = dashboard.get("relationship", {})
    living = dashboard.get("living", {})
    memory_context = dashboard.get("memory_context", {})
    prompt_guard = dashboard.get("prompt_guard", {})
    graph_fallback = dict(dashboard.get("graph_fallback") or {})

    render_runtime_rejection_banner(dismiss_key="dismiss-runtime-rejection-status")
    render_runtime_guardrails_card(dismiss_key="dismiss-runtime-guardrails-status")
    render_runtime_semantic_strip(thread_id=str(api.active_thread_id or "default"))

    st.subheader("Dad Runtime Status")
    st.caption("Live operational and emotional state — everything that matters at a glance.")

    col1, col2, col3 = st.columns(3)
    col1.metric("Health Score", f"{int(health.get('health_score', 100))}/100")
    col2.metric("Runtime Pressure", health.get("level", "green").upper())
    col3.metric("Open Threads", dashboard.get("threads", {}).get("open", 0))

    with st.container(border=True):
        st.subheader("Conversation Engine")
        if graph_fallback.get("active"):
            mode = str(graph_fallback.get("degraded_mode") or "legacy").replace("_", " ")
            count = int(graph_fallback.get("event_count", 0) or 0)
            last_error = str(graph_fallback.get("last_error") or "").strip()
            st.warning(
                f"Graph orchestration is degraded to {mode}. "
                f"Dad remains available via legacy turn processing ({count} recent event{'s' if count != 1 else ''})."
            )
            st.caption(str(graph_fallback.get("last_timestamp") or ""))
            if last_error:
                st.caption(f"Last graph error: {last_error}")
        else:
            st.success("Graph orchestration healthy. No recent fallback events.")

    with st.container(border=True):
        st.subheader("Dad's Internal State")
        internal = shell.get("internal_debug", {})
        hypotheses = len(internal.get("relationship_hypotheses", []))
        traits = len(internal.get("active_persona_traits", []))
        moods = len(internal.get("recent_moods", []))

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Hypotheses", hypotheses)
        c2.metric("Active Traits", traits)
        c3.metric("Recent Moods", moods)
        c4.metric("Learning Cycles", int(api.MEMORY_STORE.get("learning_cycle_count", 0)))

        last_learning = str(api.MEMORY_STORE.get("last_continuous_learning_at") or "Never")[:16]
        st.caption(f"Last learning cycle: {last_learning}")

    with st.container(border=True):
        st.subheader("Runtime & Memory Health")
        level = str(health.get("level", "green")).lower()
        if level == "red":
            st.error("Runtime is under heavy pressure right now.")
        elif level == "yellow":
            st.warning("Some pressure is building. Quiet mode may help.")
        else:
            st.success("Runtime is healthy and stable.")

        col_a, col_b = st.columns(2)
        _mem_tokens = int(memory_context.get("tokens", 0) or 0)
        _mem_budget = max(1, int(memory_context.get("budget_tokens", 1) or 1))
        _lucidity_pct = min(1.0, _mem_tokens / _mem_budget)
        if _lucidity_pct >= 0.90:
            _lucidity_color, _lucidity_label = "#c0392b", "Strained"
        elif _lucidity_pct >= 0.70:
            _lucidity_color, _lucidity_label = "#e67e22", "Busy"
        elif _lucidity_pct >= 0.45:
            _lucidity_color, _lucidity_label = "#27ae60", "Sharp"
        else:
            _lucidity_color, _lucidity_label = "#2980b9", "Clear"
        with col_a:
            st.markdown("**Focus (Lucidity)**")
            st.progress(_lucidity_pct, text=f"{_lucidity_label} — {int(_lucidity_pct * 100)}% context used")
            st.markdown(
                f"<span style='color:{_lucidity_color};font-size:0.82rem;'>"
                f"● {_mem_tokens:,} / {_mem_budget:,} tokens</span>",
                unsafe_allow_html=True,
            )
        col_b.metric("Prompt Trims This Session", prompt_guard.get("trim_count", 0))

    with st.container(border=True):
        st.subheader("Session Controls")
        preferences = ui_preferences()
        selected_mode = st.selectbox(
            "Power mode",
            options=["turbo", "battery"],
            index=0 if str(preferences.get("power_mode", "turbo")) == "turbo" else 1,
            format_func=lambda value: "Turbo" if value == "turbo" else "Battery",
            key="status-power-mode",
        )
        if selected_mode != str(preferences.get("power_mode", "turbo")):
            message = apply_power_mode(get_chat_event_api(), selected_mode)
            st.success(message)
            st.rerun()

        ratio = float(health.get("memory_context_ratio", 0.0) or 0.0)
        can_purge = ratio >= 0.93
        reset_mode = st.radio(
            "Reset mode",
            options=["soft", "full"],
            horizontal=True,
            format_func=lambda value: "Soft reset" if value == "soft" else "Full purge",
            key="status-reset-mode",
        )
        if st.button("Reset session context", use_container_width=True, disabled=not can_purge, key="status-reset-session"):
            result = purge_session_context(bot, mode=reset_mode)
            if result.get("mode") == "soft":
                st.success("Soft reset complete. Cleared active history while preserving one short context summary.")
            else:
                st.success("Full session purge complete. Long-term memory and internal reasoning state were preserved.")
            st.rerun()
        if not can_purge:
            st.caption("Reset unlocks automatically as context pressure rises.")

    with st.container(border=True):
        st.subheader("Relationship Meter")
        trust = int(relationship.get("trust_level", 50))
        openness = int(relationship.get("openness_level", 50))
        st.progress(trust / 100, text=f"Trust — {trust}/100")
        st.progress(openness / 100, text=f"Openness — {openness}/100")
        st.caption(f"Current hypothesis: **{relationship.get('active_hypothesis_label', 'Supportive Baseline')}**")

    with st.container(border=True):
        st.subheader("Quick Actions")
        cols = st.columns(5)
        with cols[0]:
            if st.button("Force Consolidation", use_container_width=True):
                get_chat_event_api().consolidate_memories(force=True)
                st.success("Memory consolidation complete.")
                st.rerun()
        with cols[1]:
            if st.button("Clear Semantic Index", use_container_width=True):
                get_chat_event_api().clear_semantic_memory_index()
                st.success("Semantic index cleared.")
                st.rerun()
        with cols[2]:
            if st.button("Export Memory", use_container_width=True):
                path = DEFAULT_EXPORT_PATH.with_name(f"memory_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
                api.export_memory_store(path)
                st.success(f"Exported to {path.name}")
        with cols[3]:
            if st.button("Optimize Hardware", use_container_width=True):
                get_chat_event_api().apply_hardware_optimization(confirm=True)
                st.success("Hardware optimization applied.")
        with cols[4]:
            quiet = api.health_quiet_mode_enabled()
            label = "Disable Quiet Mode" if quiet else "Enable Quiet Mode"
            if st.button(label, use_container_width=True):
                get_chat_event_api().set_health_quiet_mode(not quiet, save=True)
                st.success(f"Quiet mode {'enabled' if not quiet else 'disabled'}.")
                st.rerun()

    with st.container(border=True):
        st.subheader("Local Tools Bridge")
        mcp_status = dict(api.local_mcp_status() or {})
        running = bool(mcp_status.get("running"))
        state_text = "Running" if running else "Stopped"
        st.caption(f"dadbot-local-services status: {state_text}")
        mcp_col1, mcp_col2, mcp_col3 = st.columns(3)
        mcp_col1.metric("Tool Surface", int(mcp_status.get("tool_count", 0) or 0))
        mcp_col2.metric("Local State", int(mcp_status.get("local_state_entries", 0) or 0))
        mcp_col3.metric("Process", state_text)
        action_col1, action_col2, action_col3 = st.columns(3)
        if action_col1.button("Start MCP Server", use_container_width=True, disabled=running, key="status-mcp-start"):
            status = get_chat_event_api().start_local_mcp_server_process()
            st.success(f"Local MCP server started (pid={status.get('pid')}).")
            st.rerun()
        if action_col2.button("Restart MCP Server", use_container_width=True, key="status-mcp-restart"):
            status = get_chat_event_api().start_local_mcp_server_process(restart=True)
            st.success(f"Local MCP server restarted (pid={status.get('pid')}).")
            st.rerun()
        if action_col3.button("Stop MCP Server", use_container_width=True, disabled=not running, key="status-mcp-stop"):
            get_chat_event_api().stop_local_mcp_server_process()
            st.success("Local MCP server stopped.")
            st.rerun()
        st.caption(f"VS Code task: {mcp_status.get('task_label', 'Run Dad Bot MCP Server')}")
        st.caption(f"Stdout log: {mcp_status.get('stdout_log_path', '')}")
        st.caption(f"Stderr log: {mcp_status.get('stderr_log_path', '')}")
        log_tail = api.local_mcp_log_tail(lines=12)
        with st.expander("Recent MCP stdout", expanded=False):
            st.code(log_tail.get("stdout") or "[no stdout yet]", language="text")
        with st.expander("Recent MCP stderr", expanded=False):
            st.code(log_tail.get("stderr") or "[no stderr yet]", language="text")


@maybe_fragment
def render_sidebar(bot: DadBot):
    api = get_chat_event_api()
    if DAD_AVATAR_PATH.exists():
        st.image(str(DAD_AVATAR_PATH), width=200)
    else:
        st.markdown("<div style='font-size:6rem; text-align:center; margin:1rem 0;'>🧔</div>", unsafe_allow_html=True)
    st.caption("Dad is always here for you, Tony.")

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
            st.caption("Install the optional MCP dependency to expose Dad's local reminder, calendar, email, and state tools.")
        st.caption(f"Narrative memories distilled: {int(shell.get('narrative_memory_count', 0) or 0)}")

    with st.container(border=True):
        st.markdown("**Workshop Controls**")
        st.caption("Health, diagnostics, and tuning live in Dad's Workshop.")
        if st.button("Open Dad's Workshop", use_container_width=True, key="sidebar-open-workshop"):
            st.session_state.primary_view = "workshop"
            st.session_state["workshop-section"] = "Status"
            st.rerun()

    st.subheader("Threads")
    all_threads = list(api.list_chat_threads() or [])
    open_only = st.checkbox("Show open threads only", value=True, key="sidebar-thread-open-only")
    visible_threads = [
        thread for thread in all_threads if (not open_only or not thread.get("closed"))
    ]
    visible_threads = visible_threads or all_threads
    thread_labels = []
    label_to_thread_id = {}
    active_label = None
    for thread in visible_threads:
        thread_id = str(thread.get("thread_id") or "")
        title = str(thread.get("title") or "Chat")
        turns = int(thread.get("turn_count", 0) or 0)
        closed_suffix = " | closed" if thread.get("closed") else ""
        label = f"{title} | {turns} turns{closed_suffix}"
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
            switch_active_thread(selected_thread_id)
            st.rerun()
        selected_thread = next((t for t in visible_threads if str(t.get("thread_id") or "") == str(selected_thread_id or "")), {})
        st.caption(str(selected_thread.get("last_message") or "Fresh chat"))
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
    if nav_cols[0].button("💬 Chat", key="bottom-nav-chat", use_container_width=True, type="primary" if active_view == "chat" else "secondary"):
        state_manager.set_primary_view("chat")
        st.rerun()
    if nav_cols[1].button("🩺 Status", key="bottom-nav-status", use_container_width=True, type="primary" if active_view == "status" else "secondary"):
        state_manager.set_primary_view("status")
        st.rerun()
    if nav_cols[2].button("🛠️ Workshop", key="bottom-nav-workshop", use_container_width=True, type="primary" if active_view == "workshop" else "secondary"):
        state_manager.set_primary_view("workshop")
        st.rerun()
    if nav_cols[3].button("🎙️ Voice", key="bottom-nav-voice", use_container_width=True, type="primary" if active_view == "voice" else "secondary"):
        state_manager.set_primary_view("voice")
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)


@maybe_fragment
def render_workshop_tab(bot: DadBot):
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
        render_preferences_tab(api)
    elif section == "Data":
        render_data_tab(api)
    else:
        render_mobile_tab(bot)


def main():
    st.set_page_config(page_title="Dad Bot", page_icon="🧔", layout="centered", initial_sidebar_state="expanded")
    bot = get_runtime().bot
    initialize_session(bot)
    apply_ui_preferences(get_chat_event_api())
    update_ui_mood(bot)
    inject_custom_css(ui_preferences())
    inject_pwa_metadata(ui_preferences())
    require_pin(bot)
    render_runtime_rejection_banner(dismiss_key="dismiss-runtime-rejection-main")
    first_run_wizard(bot)
    active_thread = get_chat_event_api().active_chat_thread() or create_new_thread()

    state_manager.active_primary_view()

    with st.sidebar:
        render_sidebar(bot)

    nav_options = ["chat", "status", "workshop", "voice"]
    current_nav = state_manager.active_primary_view()
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

    if nav_choice == "chat":
        render_chat_tab(bot, active_thread)
    elif nav_choice == "status":
        render_status_tab(bot)
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


if __name__ == "__main__":
    main()
