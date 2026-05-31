"""Shared UI utility helpers that depend on streamlit + external libs only.

These functions are safe to import from any dadbot.ui submodule because they
never import from dad_streamlit (no circular deps).
"""

from __future__ import annotations

import re
import shutil
from datetime import datetime
from typing import TYPE_CHECKING
from urllib.error import URLError
from urllib.request import urlopen

import ollama
import streamlit as st

from dadbot.ui.prefs_state import ui_preferences, voice_preferences

if TYPE_CHECKING:
    from dadbot.core.dadbot import DadBot

# Optional STT backend detection â€” mirrors dad_streamlit optional imports
try:
    from faster_whisper import WhisperModel  # type: ignore[import-not-found]
except Exception:
    WhisperModel = None

try:
    import whisper as openai_whisper  # type: ignore[import-not-found]
except Exception:
    openai_whisper = None

try:
    import pyttsx3  # type: ignore[import-not-found]
except Exception:
    pyttsx3 = None

try:
    import edge_tts  # type: ignore[import-not-found]
except Exception:
    edge_tts = None

__all__ = [
    "apply_power_mode",
    "apply_ui_preferences",
    "export_bundle_payload",
    "fetch_ical_events",
    "find_available_image_model",
    "heritage_files_with_limits",
    "local_stt_backend_status",
    "local_tts_backend_status",
    "render_voice_dependency_help",
    "voice_profile_catalog",
]

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB


def voice_profile_catalog() -> dict:
    return {
        "warm_dad": "Balanced and reassuring. Best default for day-to-day chats.",
        "deep_dad": "Lower and steadier tone for grounding or coaching moments.",
        "gentle_dad": "Softer and calmer tone for stressful or emotional check-ins.",
    }


def render_voice_dependency_help(*, context_key: str) -> None:
    st.markdown(
        "Voice features use optional local packages. Dad can still chat without them, "
        "but STT/TTS needs: "
        "[faster-whisper](https://pypi.org/project/faster-whisper/) or "
        "[openai-whisper](https://pypi.org/project/openai-whisper/) for transcription, and "
        "[edge-tts](https://pypi.org/project/edge-tts/), [pyttsx3](https://pypi.org/project/pyttsx3/), "
        "or [Piper](https://github.com/rhasspy/piper) for speech.",
    )

    stt_ok, stt_backend, stt_status = local_stt_backend_status(None)
    tts_ok, tts_backend, tts_status = local_tts_backend_status()
    status_col1, status_col2 = st.columns(2)
    status_col1.caption("Speech-to-text (STT)")
    (status_col1.success if stt_ok else status_col1.warning)(f"{stt_backend}: {stt_status}")
    status_col2.caption("Text-to-speech (TTS)")
    (status_col2.success if tts_ok else status_col2.warning)(f"{tts_backend}: {tts_status}")

    st.caption("Fastest install (recommended):")
    st.code("python install.py --with-voice", language="shell")
    st.caption("Or install just the extras:")
    st.code("pip install .[voice]", language="shell")
    st.caption("After installing, restart the app and try voice again.")


def local_stt_backend_status(preferences) -> tuple[bool, str, str]:
    voice = voice_preferences() if preferences is None else preferences
    if not bool(voice.get("stt_enabled", True)):
        return False, "none", "Local STT is disabled in voice preferences."
    requested_backend = str(voice.get("stt_backend") or "auto").strip().lower()
    if requested_backend == "faster_whisper":
        if WhisperModel is None:
            return False, "none", "faster-whisper is not installed."
        return True, "faster_whisper", "Using local faster-whisper STT."
    if requested_backend == "openai_whisper":
        if openai_whisper is None:
            return False, "none", "openai-whisper is not installed."
        return True, "openai_whisper", "Using local openai-whisper STT."
    if WhisperModel is not None:
        return True, "faster_whisper", "Using local faster-whisper STT."
    if openai_whisper is not None:
        return True, "openai_whisper", "Using local openai-whisper STT."
    return (
        False,
        "none",
        "Install faster-whisper or openai-whisper for local transcription.",
    )


def local_tts_backend_status(preferences: dict | None = None) -> tuple[bool, str, str]:
    voice = voice_preferences() if preferences is None else preferences
    requested = str(voice.get("tts_backend") or "auto").strip().lower()

    piper_model_path = str(voice.get("tts_piper_model_path") or "").strip()
    piper_ready = bool(shutil.which("piper") and piper_model_path)
    edge_ready = edge_tts is not None
    pyttsx3_ready = pyttsx3 is not None

    if requested == "piper":
        if piper_ready:
            return True, "piper", "Using Piper neural TTS."
        return False, "none", "Piper selected, but executable or model path is missing."

    if requested == "edge_tts":
        if edge_ready:
            return True, "edge_tts", "Using edge-tts neural voice."
        return False, "none", "edge-tts selected, but package is not installed."

    if requested == "auto":
        if piper_ready:
            return True, "piper", "Auto-selected Piper neural TTS."
        if edge_ready:
            return True, "edge_tts", "Auto-selected edge-tts neural voice."
        if pyttsx3_ready:
            return True, "pyttsx3", "Auto-selected pyttsx3 system voice."
        return False, "none", "No TTS backend found. Install edge-tts, pyttsx3, or Piper."

    if pyttsx3_ready:
        return True, "pyttsx3", "Using local pyttsx3 text-to-speech."
    return False, "none", "pyttsx3 selected, but package is not installed."


def apply_power_mode(bot: DadBot, mode: str) -> str:
    normalized_mode = str(mode or "turbo").strip().lower()
    preferences = ui_preferences()
    if normalized_mode == "battery":
        preferences["power_mode"] = "battery"
        preferences["light_mode"] = True
        bot.LIGHT_MODE = True
        bot.set_health_quiet_mode(True, save=False)
        bot.apply_hardware_optimization(confirm=True)
        return "Battery mode enabled"
    preferences["power_mode"] = "turbo"
    preferences["light_mode"] = False
    bot.LIGHT_MODE = False
    bot.set_health_quiet_mode(False, save=False)
    return "Turbo mode enabled"


def apply_ui_preferences(bot: DadBot) -> None:
    preferences = ui_preferences()
    power_mode = str(preferences.get("power_mode") or "turbo").strip().lower()
    if power_mode == "battery":
        preferences["light_mode"] = True
    bot.APPEND_SIGNOFF = bool(preferences.get("append_signoff", True))
    bot.LIGHT_MODE = bool(preferences.get("light_mode", False))


@st.cache_data(show_spinner=False, ttl=120)
def find_available_image_model(candidates: tuple[str, ...]) -> str | None:
    for candidate in candidates:
        try:
            ollama.show(candidate)
            return candidate
        except Exception:
            continue
    return None


def fetch_ical_events(url: str, max_events: int = 10) -> tuple[list[dict], str]:
    """Fetch and parse upcoming events from a public iCal/ICS feed URL.
    Returns (events_list, error_message). No external deps â€” pure stdlib.
    """
    events: list[dict] = []
    try:
        with urlopen(url, timeout=8) as response:
            raw = response.read().decode("utf-8", errors="replace")
    except URLError as exc:
        return [], f"Could not fetch calendar feed: {exc}"
    except Exception as exc:
        return [], f"Calendar fetch error: {exc}"

    vevent_pattern = re.compile(r"BEGIN:VEVENT(.*?)END:VEVENT", re.DOTALL)
    for match in vevent_pattern.finditer(raw):
        block = match.group(1)

        def _prop(name: str) -> str:
            m = re.search(rf"^{name}[;:][^\r\n]*[:\n]([^\r\n]*)", block, re.MULTILINE)
            if not m:
                m = re.search(rf"^{name}:([^\r\n]*)", block, re.MULTILINE)
            return (m.group(1).strip() if m else "").replace("\\n", " ").replace("\\,", ",")

        dtstart_raw = _prop("DTSTART")
        dtend_raw = _prop("DTEND")
        summary = _prop("SUMMARY") or "(no title)"
        location = _prop("LOCATION")
        description = _prop("DESCRIPTION")

        def _format_dt(raw_dt: str) -> str:
            raw_dt = re.sub(r"T.*", "", raw_dt[:8])
            try:
                from datetime import date as _dc

                return _dc(
                    int(raw_dt[:4]),
                    int(raw_dt[4:6]),
                    int(raw_dt[6:8]),
                ).isoformat()
            except Exception:
                return raw_dt

        event_date = _format_dt(dtstart_raw) if dtstart_raw else ""
        try:
            from datetime import date as _dc2

            if event_date and _dc2.fromisoformat(event_date) < _dc2.today():
                continue
        except Exception:
            event_date = event_date

        events.append(
            {
                "summary": summary,
                "start": event_date,
                "end": _format_dt(dtend_raw) if dtend_raw else "",
                "location": location,
                "description": description[:120] if description else "",
            },
        )
        if len(events) >= max_events:
            break

    events.sort(key=lambda e: e.get("start") or "")
    return events, ""


def export_bundle_payload(bot: DadBot) -> dict:
    snapshot = bot.snapshot_session_state()
    return {
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "profile": bot.PROFILE,
        "memory_store": snapshot.get("memory_store", {}),
        "chat_threads": snapshot.get("chat_threads", []),
        "active_thread_id": snapshot.get("active_thread_id"),
        "thread_snapshots": snapshot.get("thread_snapshots", {}),
        "ui_preferences": dict(ui_preferences()),
    }


def heritage_files_with_limits(uploaded_files) -> tuple[list, list[str]]:
    """Filter uploaded files, enforcing the MAX_UPLOAD_BYTES size limit.

    Returns (accepted_files, issues) where issues is a list of warning strings.
    """
    accepted = []
    issues: list[str] = []
    for uploaded_file in list(uploaded_files or []):
        try:
            raw = uploaded_file.getvalue() or b""
        except Exception as exc:
            issues.append(
                f"{getattr(uploaded_file, 'name', 'upload')}: could not read file ({exc}).",
            )
            continue
        if not raw:
            issues.append(
                f"{getattr(uploaded_file, 'name', 'upload')}: file was empty.",
            )
            continue
        if len(raw) > MAX_UPLOAD_BYTES:
            issues.append(
                f"{getattr(uploaded_file, 'name', 'upload')}: file is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)}MB limit.",
            )
            continue
        accepted.append(uploaded_file)
    return accepted, issues
