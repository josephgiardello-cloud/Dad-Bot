from __future__ import annotations

import base64
import logging
import os
import uuid
from typing import Any, Callable

import ollama
import streamlit as st

from dadbot.runtime_core.streamlit_runtime import StreamlitRuntime
from dadbot.ui.helpers import apply_ui_preferences, find_available_image_model
from dadbot.ui.prefs_state import sync_ui_voice_from_profile, ui_preferences
from dadbot.ui.state_manager import record_runtime_rejection

logger = logging.getLogger(__name__)


@st.cache_resource(show_spinner=False)
def get_runtime() -> StreamlitRuntime:
    return StreamlitRuntime.build()


@st.cache_resource(show_spinner=False)
def get_chat_event_api() -> Any:
    return get_runtime().api


def process_prompt_via_runtime(*, thread_id: str, prompt: str, attachments: list[dict] | None = None) -> dict:
    runtime = get_runtime()
    return runtime.send_user_message(thread_id=thread_id, content=str(prompt or "").strip(), attachments=attachments)


def emit_voice_runtime_ledger_event(event_type: str, payload: dict) -> None:
    api = get_chat_event_api()
    event_name = str(event_type or "VOICE_EVENT").strip().upper() or "VOICE_EVENT"
    if event_name not in {"VOICE_STATE_TRANSITION", "VOICE_EVENT"}:
        return
    merged_payload = dict(payload or {})
    trace_id = str(merged_payload.get("trace_id") or st.session_state.setdefault("voice_trace_id", uuid.uuid4().hex))
    session_id = str(merged_payload.get("session_id") or api.active_thread_id or "voice-ui")
    try:
        api.emit_voice_runtime_ledger_event(
            event_type=event_name,
            session_id=session_id,
            trace_id=trace_id,
            kernel_step_id="voice.control_plane",
            payload=merged_payload,
        )
    except Exception as exc:
        record_runtime_rejection(exc, action="emit_voice_runtime_ledger_event")
        logger.debug("Voice ledger emission skipped for %s: %s", event_name, exc)


def purge_session_context(*, bot: Any, initialize_session: Callable[[Any], None], mode: str = "full") -> dict:
    api = get_chat_event_api()
    normalized_mode = str(mode or "full").strip().lower()
    if normalized_mode == "soft":
        result = api.soft_reset_session_context(preserve_recent_summary=True)
    else:
        api.reset_session_state()
        result = {"mode": "full", "preserved_summary": ""}
    st.session_state.active_thread_id = api.active_thread_id
    initialize_session(bot)
    return result


def ui_shell_snapshot(bot: Any = None) -> dict:
    _ = bot
    return get_chat_event_api().ui_shell_snapshot()


def default_thread_messages(initial_greeting: str) -> list[dict]:
    return [{"role": "assistant", "content": get_chat_event_api().opening_message(initial_greeting)}]


def bot_messages_for_thread(thread_id: str, *, default_greeting: str) -> list[dict]:
    return get_chat_event_api().snapshot_thread_messages(thread_id, default_greeting=default_greeting)


def initialize_session(bot: Any, *, default_export_path: str, initial_greeting: str) -> None:
    api = get_chat_event_api()
    api.ensure_chat_thread_state()
    api.sync_active_thread_snapshot()
    if "active_thread_id" not in st.session_state:
        st.session_state.active_thread_id = api.active_thread_id
    if "export_path" not in st.session_state:
        st.session_state.export_path = str(default_export_path)
    if st.session_state.active_thread_id != api.active_thread_id:
        api.switch_chat_thread(st.session_state.active_thread_id)
    if not st.session_state.get("ui_profile_seeded", False):
        sync_ui_voice_from_profile(bot)
        st.session_state.ui_profile_seeded = True
    if api.active_thread_id is None:
        create_new_thread(initial_greeting=initial_greeting)


def switch_active_thread(thread_id: str) -> None:
    api = get_chat_event_api()
    api.switch_chat_thread(thread_id)
    st.session_state.active_thread_id = api.active_thread_id


def create_new_thread(*, initial_greeting: str) -> dict:
    api = get_chat_event_api()
    thread = api.create_chat_thread()
    st.session_state.active_thread_id = thread["thread_id"]
    api.seed_thread(thread["thread_id"], default_thread_messages(initial_greeting))
    return thread


def optimize_runtime_for_hardware() -> dict:
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


def generate_dad_photo() -> bytes | None:
    try:
        candidates = ["flux", "flux-dev", "flux-schnell", "sdxl", "stable-diffusion"]
        model = find_available_image_model(tuple(candidates))
        if not model:
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
        return images[0] if images else None
    except Exception:
        logger.exception("Photo generation failed")
        return None


def emit_generated_photo_message(*, thread_id: str, message: str = "Here's a quick photo I took for you, buddy. Love you.") -> bool:
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