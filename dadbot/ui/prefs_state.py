"""Streamlit session-state helpers for UI preferences and voice settings.

These functions are pure in the sense that they never import from dad_streamlit,
only from streamlit and the DadBot type annotation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import streamlit as st

if TYPE_CHECKING:
    from dadbot.core.dadbot import DadBot

__all__ = [
    "_voice_defaults",
    "default_ui_preferences",
    "notification_settings",
    "profile_voice_preferences",
    "sync_ui_voice_from_profile",
    "ui_preferences",
    "voice_preferences",
    "voice_profile_catalog",
]


def default_ui_preferences() -> dict:
    return {
        "theme_mode": "warm",
        "power_mode": "turbo",
        "auto_mood_theme": True,
        "font_scale": 1.0,
        "high_contrast": False,
        "append_signoff": True,
        "light_mode": False,
        "voice": {
            "enabled": False,
            "mode": "push_to_talk",
            "auto_send_always_listening": True,
            "wake_word_required": False,
            "wake_word_phrase": "hey dad",
            "auto_listen_allowed": False,
            "last_mode": "push_to_talk",
            "stt_enabled": True,
            "stt_backend": "auto",
            "stt_model": "base",
            "stt_language": "en",
            "tts_enabled": True,
            "tts_autoplay": False,
            "tts_voice": "warm_dad",
            "tts_backend": "pyttsx3",
            "tts_piper_model_path": "",
            "tts_rate": 0,
            "warmth": 70,
            "dad_joke_frequency": 35,
            "pacing": 50,
            "muted": False,
            "mic_preference": "default",
            "last_used_device": "default",
            "interruptions_enabled": True,
            "barge_in_enabled": True,
            "barge_in_min_audio_bytes": 4000,
            "allow_tts_cancel": True,
            "priority_override_enabled": True,
            "known_device_ids": ["default"],
        },
    }


def _voice_defaults() -> dict:
    """Return a fresh copy of the default voice preference block."""
    return dict(default_ui_preferences()["voice"])


def ui_preferences() -> dict:
    defaults = default_ui_preferences()
    if "ui_preferences" not in st.session_state:
        st.session_state.ui_preferences = defaults
        return st.session_state.ui_preferences

    merged = dict(defaults)
    current = dict(st.session_state.ui_preferences or {})
    merged.update({key: value for key, value in current.items() if key != "voice"})

    default_voice = dict(defaults["voice"])
    current_voice = current.get("voice") if isinstance(current.get("voice"), dict) else {}
    default_voice.update(current_voice)
    merged["voice"] = default_voice

    st.session_state.ui_preferences = merged
    return st.session_state.ui_preferences


def voice_preferences() -> dict:
    preferences = ui_preferences()
    voice = preferences.get("voice")
    if not isinstance(voice, dict):
        voice = _voice_defaults()
        preferences["voice"] = voice
    return voice


def profile_voice_preferences(bot: DadBot) -> dict:
    """Return persisted voice preferences merged over defaults."""
    defaults = _voice_defaults()
    profile_voice = bot.PROFILE.get("voice", {}) if isinstance(bot.PROFILE, dict) else {}
    if isinstance(profile_voice, dict):
        defaults.update(profile_voice)
    return defaults


def sync_ui_voice_from_profile(bot: DadBot) -> None:
    """Keep session UI cache aligned with profile-backed persisted voice settings."""
    preferences = ui_preferences()
    preferences["voice"] = dict(profile_voice_preferences(bot))


def notification_settings(bot: DadBot) -> dict:
    configured = bot.PROFILE.get("notifications", {}) if isinstance(bot.PROFILE, dict) else {}
    if not isinstance(configured, dict):
        configured = {}
    backend = str(configured.get("backend") or "auto").strip().lower() or "auto"
    if backend not in {"auto", "notifypy", "plyer"}:
        backend = "auto"
    try:
        quiet_start = int(configured.get("quiet_hours_start", 23))
    except (TypeError, ValueError):
        quiet_start = 23
    try:
        quiet_end = int(configured.get("quiet_hours_end", 7))
    except (TypeError, ValueError):
        quiet_end = 7
    return {
        "enabled": bool(configured.get("enabled", False)),
        "backend": backend,
        "quiet_hours_start": max(0, min(23, quiet_start)),
        "quiet_hours_end": max(0, min(23, quiet_end)),
        "notify_patterns": bool(configured.get("notify_patterns", True)),
        "notify_reminders": bool(configured.get("notify_reminders", True)),
    }


def voice_profile_catalog() -> dict:
    return {
        "warm_dad": "Balanced and reassuring. Best default for day-to-day chats.",
        "deep_dad": "Lower and steadier tone for grounding or coaching moments.",
        "gentle_dad": "Softer and calmer tone for stressful or emotional check-ins.",
    }
