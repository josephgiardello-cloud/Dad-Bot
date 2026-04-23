"""Preferences surface renderer."""

from __future__ import annotations

import base64
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import ollama
import streamlit as st

from dadbot.ui.helpers import (
    apply_power_mode,
    apply_ui_preferences,
    find_available_image_model,
    local_stt_backend_status,
    local_tts_backend_status,
    render_voice_dependency_help,
    voice_profile_catalog,
)
from dadbot.ui.prefs_state import notification_settings, profile_voice_preferences, ui_preferences
from dadbot.ui.utils import option_index

if TYPE_CHECKING:
    from dadbot.core.dadbot import DadBot

_STATIC_DIR = Path("static")
DAD_AVATAR_PATH = _STATIC_DIR / "dad_avatar.png"

__all__ = ["render_preferences_tab"]


def render_preferences_tab(bot: "DadBot") -> None:
    preferences = ui_preferences()
    voice = profile_voice_preferences(bot)
    runtime = bot.runtime_settings()
    cadence = bot.cadence_settings()
    tools = bot.agentic_tool_settings()
    calibration = bot.relationship_calibration_settings()
    notifications = notification_settings(bot)
    llm_profile = bot.PROFILE.get("llm", {}) if isinstance(bot.PROFILE, dict) else {}
    current_llm_provider = str(llm_profile.get("provider") or getattr(bot, "LLM_PROVIDER", "ollama") or "ollama").strip().lower()
    current_llm_model = str(llm_profile.get("model") or getattr(bot, "LLM_MODEL", bot.MODEL_NAME) or bot.MODEL_NAME).strip()
    preset_catalog = bot.persona_preset_catalog()
    with st.container(border=True):
        st.subheader("Persona")
        st.caption("Tune how Dad sounds and what he emphasizes.")
        with st.form("persona-settings"):
            preset_options = list(preset_catalog.keys())
            selected_preset = st.selectbox("Persona preset", options=preset_options, index=preset_options.index(bot.current_persona_preset()), format_func=lambda key: preset_catalog[key]["label"])
            dad_name = st.text_input("Dad display name", value=bot.STYLE.get("name", "Dad"))
            listener_name = st.text_input("Listener name", value=bot.STYLE.get("listener_name", "Tony"))
            signoff = st.text_input("Signoff", value=bot.STYLE.get("signoff", "Love you, buddy."))
            behavior_rules = st.text_area("Behavior rules", value="\n".join(bot.STYLE.get("behavior_rules", [])), height=180)
            opening_messages = st.text_area("Opening message options", value="\n".join(bot.opening_message_candidates()), height=110)
            if st.form_submit_button("Save persona", type="primary"):
                bot.update_style_profile(name=dad_name, listener_name=listener_name, signoff=signoff, behavior_rules=behavior_rules.splitlines(), persona_preset=selected_preset, save=False)
                bot.apply_persona_preset(selected_preset, save=False)
                bot.update_style_profile(name=dad_name, listener_name=listener_name, signoff=signoff, behavior_rules=behavior_rules.splitlines(), persona_preset=selected_preset, save=False)
                bot.update_opening_messages_profile(opening_messages.splitlines(), save=False)
                bot.save_profile()
                st.success("Persona settings saved.")
                st.rerun()
    with st.container(border=True):
        st.subheader("Avatar Studio")
        st.caption("Generate and save a custom photo of Dad using any Ollama image model. Stored as the persistent sidebar avatar.")
        _avatar_prompt = st.text_area(
            "Avatar generation prompt",
            value=(
                "Photorealistic warm portrait of a friendly 56-year-old father with kind eyes, "
                "short neatly trimmed graying hair, gentle reassuring smile, wearing a soft flannel shirt, "
                "standing in a cozy home kitchen with wooden cabinets and soft natural window light, "
                "heartwarming atmosphere, high detail, cinematic lighting"
            ),
            height=110,
            key="avatar-studio-prompt",
        )
        _avatar_col1, _avatar_col2 = st.columns([1, 2])
        with _avatar_col1:
            if DAD_AVATAR_PATH.exists():
                st.image(str(DAD_AVATAR_PATH), caption="Current avatar", use_container_width=True)
            else:
                st.markdown("<div style='font-size:5rem;text-align:center;'>🧔</div>", unsafe_allow_html=True)
                st.caption("No avatar yet")
        with _avatar_col2:
            _img_model = None
            _img_candidates = ["flux", "flux-dev", "flux-schnell", "sdxl", "stable-diffusion"]
            _img_model = find_available_image_model(tuple(_img_candidates))
            if _img_model:
                st.success(f"Image model ready: **{_img_model}**")
            else:
                st.warning("No Ollama image model found. Pull `flux` or `sdxl` in Ollama to enable generation.")
                st.code("ollama pull flux", language="bash")
            if st.button("Generate new avatar", use_container_width=True, disabled=not _img_model, type="primary"):
                with st.spinner(f"Generating avatar with {_img_model}..."):
                    try:
                        _resp = ollama.generate(
                            model=_img_model,
                            prompt=str(_avatar_prompt or "").strip(),
                            options={"num_predict": 1},
                        )
                        _imgs = [base64.b64decode(img) for img in _resp.get("images", []) if img]
                        if _imgs:
                            DAD_AVATAR_PATH.write_bytes(_imgs[0])
                            st.success("Avatar saved! The sidebar will update on next reload.")
                            st.image(_imgs[0], caption="New avatar", use_container_width=True)
                        else:
                            st.error("Model returned no image. Try a different model or prompt.")
                    except Exception as _exc:
                        st.error(f"Generation failed: {_exc}")
            if DAD_AVATAR_PATH.exists():
                if st.button("Remove avatar (revert to emoji)", use_container_width=True):
                    try:
                        DAD_AVATAR_PATH.unlink()
                        st.success("Avatar removed.")
                        st.rerun()
                    except Exception as _exc:
                        st.error(f"Could not remove avatar: {_exc}")
    with st.container(border=True):
        st.subheader("Preferences")
        st.caption("Blend session-level UI controls with persistent Dad runtime settings.")
        with st.form("runtime-settings"):
            col1, col2 = st.columns(2)
            with col1:
                theme_mode = st.selectbox("Theme", options=["warm", "night"], index=["warm", "night"].index(preferences["theme_mode"]), format_func=lambda value: "Warm Day" if value == "warm" else "Night Shift")
                auto_mood_theme = st.checkbox("Mood visualizer (auto theme)", value=bool(preferences.get("auto_mood_theme", True)))
                power_mode = st.selectbox(
                    "Power mode",
                    options=["turbo", "battery"],
                    index=0 if str(preferences.get("power_mode", "turbo")) == "turbo" else 1,
                    format_func=lambda value: "Turbo" if value == "turbo" else "Battery",
                )
                font_scale = st.slider("Font scale", min_value=0.9, max_value=1.3, value=float(preferences["font_scale"]), step=0.05)
                high_contrast = st.checkbox("High contrast", value=bool(preferences["high_contrast"]))
                append_signoff = st.checkbox("Append signoff", value=bool(preferences["append_signoff"]))
                light_mode = st.checkbox("Light runtime mode", value=bool(preferences["light_mode"]))
                voice_enabled = st.checkbox("Enable voice interaction", value=bool(voice.get("enabled", False)))
                voice_mode = st.selectbox(
                    "Voice mode",
                    options=["push_to_talk", "always_listening", "ambient"],
                    index=option_index(["push_to_talk", "always_listening", "ambient"], str(voice.get("mode") or "push_to_talk"), fallback=0),
                    format_func=lambda value: {
                        "push_to_talk": "Push-to-talk",
                        "always_listening": "Always-listening",
                        "ambient": "Ambient (advanced)",
                    }[value],
                )
                auto_send = st.checkbox(
                    "Auto-send in always-listening",
                    value=bool(voice.get("auto_send_always_listening", True)),
                    disabled=voice_mode != "always_listening",
                )
                wake_word_required = st.checkbox(
                    "Require wake phrase in always-listening",
                    value=bool(voice.get("wake_word_required", False)),
                    disabled=voice_mode != "always_listening",
                )
                wake_word_phrase = st.text_input(
                    "Wake phrase",
                    value=str(voice.get("wake_word_phrase") or "hey dad"),
                    disabled=not wake_word_required or voice_mode != "always_listening",
                )
            with col2:
                llm_provider = st.selectbox(
                    "LLM provider",
                    options=["ollama", "openai", "anthropic", "groq", "google", "xai"],
                    index=option_index(["ollama", "openai", "anthropic", "groq", "google", "xai"], current_llm_provider, fallback=0),
                    format_func=lambda value: "Ollama (Local)" if value == "ollama" else value.upper(),
                )
                llm_model = st.text_input(
                    "LLM model",
                    value=current_llm_model,
                    help="Examples: llama3.2, gpt-4o-mini, claude-3-5-sonnet-20240620, llama3-70b",
                )
                _llm_key_map = {
                    "openai": "OPENAI_API_KEY",
                    "anthropic": "ANTHROPIC_API_KEY",
                    "groq": "GROQ_API_KEY",
                    "google": "GOOGLE_API_KEY",
                    "xai": "XAI_API_KEY",
                }
                if llm_provider == "ollama":
                    st.caption("Local Ollama — no API key required.")
                else:
                    _required_key = _llm_key_map.get(llm_provider, "")
                    if _required_key and os.environ.get(_required_key):
                        st.caption(f"✅ `{_required_key}` detected.")
                    elif _required_key:
                        st.caption(f"⚠️ Set `{_required_key}` in your environment before saving.")
                agentic_enabled = st.checkbox("Agentic tools enabled", value=tools["enabled"])
                auto_reminders = st.checkbox("Auto reminders", value=tools["auto_reminders"])
                auto_web_lookup = st.checkbox("Auto web lookup", value=tools["auto_web_lookup"])
                notify_enabled = st.checkbox("Desktop proactive notifications", value=bool(notifications.get("enabled", False)))
                notification_backend = st.selectbox(
                    "Notification backend",
                    options=["auto", "notifypy", "plyer"],
                    index=option_index(["auto", "notifypy", "plyer"], str(notifications.get("backend") or "auto"), fallback=0),
                    format_func=lambda value: value.title(),
                    disabled=not notify_enabled,
                )
                notify_reminders = st.checkbox(
                    "Notify reminder nudges",
                    value=bool(notifications.get("notify_reminders", True)),
                    disabled=not notify_enabled,
                )
                notify_patterns = st.checkbox(
                    "Notify pattern check-ins",
                    value=bool(notifications.get("notify_patterns", True)),
                    disabled=not notify_enabled,
                )
                quiet_hours_start = st.slider(
                    "Notification quiet start hour",
                    min_value=0,
                    max_value=23,
                    value=int(notifications.get("quiet_hours_start", 23) or 23),
                    disabled=not notify_enabled,
                )
                quiet_hours_end = st.slider(
                    "Notification quiet end hour",
                    min_value=0,
                    max_value=23,
                    value=int(notifications.get("quiet_hours_end", 7) or 7),
                    disabled=not notify_enabled,
                )
                max_thinking_time_seconds = st.slider("Max thinking time (seconds)", min_value=15, max_value=120, value=int(runtime["max_thinking_time_seconds"]), step=5)
                stream_max_chars = st.slider("Reply stream budget", min_value=4000, max_value=24000, value=int(runtime["stream_max_chars"]), step=1000)
                stt_enabled = st.checkbox("Enable local STT", value=bool(voice.get("stt_enabled", True)))
                stt_backend = st.selectbox(
                    "Preferred STT backend",
                    options=["auto", "faster_whisper", "openai_whisper"],
                    index=option_index(["auto", "faster_whisper", "openai_whisper"], str(voice.get("stt_backend") or "auto"), fallback=0),
                    format_func=lambda value: value.replace("_", " ").title(),
                )
                stt_model = st.selectbox(
                    "STT model",
                    options=["tiny", "base", "small"],
                    index=option_index(["tiny", "base", "small"], str(voice.get("stt_model") or "base"), fallback=1),
                )
                tts_enabled = st.checkbox("Enable local TTS", value=bool(voice.get("tts_enabled", True)))
                _piper_avail = bool(shutil.which("piper"))
                tts_backend_pref = st.selectbox(
                    "TTS backend",
                    options=["pyttsx3", "piper"],
                    index=option_index(["pyttsx3", "piper"], str(voice.get("tts_backend") or "pyttsx3"), fallback=0),
                    format_func=lambda v: "Piper (neural, high-quality)" if v == "piper" else "pyttsx3 (system voices)",
                    disabled=not tts_enabled,
                    help="Piper requires the piper executable on PATH and a .onnx model file.",
                )
                if tts_backend_pref == "piper":
                    if _piper_avail:
                        st.success("Piper executable detected on PATH.")
                    else:
                        st.warning("`piper` not found on PATH. Install from https://github.com/rhasspy/piper/releases")
                    tts_piper_model_path = st.text_input(
                        "Piper model path (.onnx)",
                        value=str(voice.get("tts_piper_model_path") or ""),
                        help="Full path to a downloaded Piper .onnx model file. Download from https://rhasspy.github.io/piper-samples/",
                        disabled=not tts_enabled,
                    )
                    st.caption("Tip: `en_US-lessac-medium.onnx` is a good starting point for an American male voice.")
                else:
                    tts_piper_model_path = str(voice.get("tts_piper_model_path") or "")
                profile_catalog = voice_profile_catalog()
                tts_voice = st.selectbox(
                    "Dad voice profile",
                    options=list(profile_catalog.keys()),
                    index=option_index(list(profile_catalog.keys()), str(voice.get("tts_voice") or "warm_dad"), fallback=0),
                    format_func=lambda value: value.replace("_", " ").title(),
                )
                st.caption(profile_catalog.get(tts_voice, ""))
                tts_autoplay = st.checkbox("Autoplay Dad voice replies", value=bool(voice.get("tts_autoplay", False)), disabled=not tts_enabled)
                tts_rate = st.slider("Dad voice speed", min_value=-40, max_value=40, value=int(voice.get("tts_rate", 0)), step=5, disabled=not tts_enabled)
                warmth = st.slider("Warmth", min_value=0, max_value=100, value=int(voice.get("warmth", 70)), step=5)
                dad_joke_frequency = st.slider("Dad-joke frequency", min_value=0, max_value=100, value=int(voice.get("dad_joke_frequency", 35)), step=5)
                pacing = st.slider("Voice pacing", min_value=0, max_value=100, value=int(voice.get("pacing", 50)), step=5)
                st.caption("Performance note: first local STT run may take 10-30 seconds while Whisper model weights load.")
                stt_ready, _stt_backend_active, stt_message = local_stt_backend_status(
                    {
                        "stt_enabled": bool(stt_enabled),
                        "stt_backend": stt_backend,
                    }
                )
                _tts_ready, _tts_backend_active, tts_message = local_tts_backend_status()
                if not stt_ready:
                    st.warning(stt_message)
                    render_voice_dependency_help(context_key="prefs-stt")
                if tts_enabled and not _tts_ready:
                    st.warning(tts_message)
                    render_voice_dependency_help(context_key="prefs-tts")
            preferred_models = st.text_input("Preferred embedding models", value=", ".join(runtime["preferred_embedding_models"]))
            family_echo_turn_interval = st.slider("Family echo cadence", min_value=1, max_value=12, value=int(cadence["family_echo_turn_interval"]))
            wisdom_turn_interval = st.slider("Wisdom cadence", min_value=1, max_value=12, value=int(cadence["wisdom_turn_interval"]))
            life_pattern_queue_limit = st.slider("Proactive queue limit", min_value=1, max_value=6, value=int(cadence["life_pattern_queue_limit"]))
            calibration_enabled = st.checkbox("Gentle pushback enabled", value=calibration["enabled"])
            opening_line = st.text_area("Pushback opening line", value=calibration["opening_line"], height=90)
            if st.form_submit_button("Save preferences", type="primary"):
                preferences.update(
                    {
                        "theme_mode": theme_mode,
                        "auto_mood_theme": bool(auto_mood_theme),
                        "power_mode": power_mode,
                        "font_scale": font_scale,
                        "high_contrast": high_contrast,
                        "append_signoff": append_signoff,
                        "light_mode": light_mode,
                    }
                )
                apply_power_mode(bot, power_mode)
                updated_voice = {
                    "enabled": bool(voice_enabled),
                    "mode": voice_mode,
                    "auto_send_always_listening": bool(auto_send),
                    "wake_word_required": bool(wake_word_required),
                    "wake_word_phrase": str(wake_word_phrase or "hey dad").strip().lower() or "hey dad",
                    "stt_enabled": bool(stt_enabled),
                    "stt_backend": stt_backend,
                    "stt_model": stt_model,
                    "stt_language": str(voice.get("stt_language") or "en"),
                    "tts_enabled": bool(tts_enabled),
                    "tts_autoplay": bool(tts_autoplay),
                    "tts_voice": tts_voice,
                    "tts_backend": tts_backend_pref,
                    "tts_piper_model_path": str(tts_piper_model_path or "").strip(),
                    "tts_rate": int(tts_rate),
                    "warmth": int(warmth),
                    "dad_joke_frequency": int(dad_joke_frequency),
                    "pacing": int(pacing),
                    "muted": bool(voice.get("muted", False)),
                    "mic_preference": str(voice.get("mic_preference") or "default"),
                    "last_used_device": str(voice.get("last_used_device") or "default"),
                    "auto_listen_allowed": bool(voice_mode in {"always_listening", "ambient"}),
                    "last_mode": voice_mode,
                    "interruptions_enabled": bool(voice.get("interruptions_enabled", True)),
                    "barge_in_enabled": bool(voice.get("barge_in_enabled", True)),
                    "barge_in_min_audio_bytes": int(voice.get("barge_in_min_audio_bytes", 4000) or 4000),
                    "allow_tts_cancel": bool(voice.get("allow_tts_cancel", True)),
                    "priority_override_enabled": bool(voice.get("priority_override_enabled", True)),
                    "known_device_ids": list(voice.get("known_device_ids") or ["default"]),
                }
                bot.PROFILE["voice"] = dict(updated_voice)
                preferences["voice"] = dict(updated_voice)
                apply_ui_preferences(bot)
                bot.PROFILE["llm"] = {
                    "provider": str(llm_provider or "ollama").strip().lower() or "ollama",
                    "model": str(llm_model or bot.MODEL_NAME).strip() or bot.MODEL_NAME,
                }
                bot.LLM_PROVIDER = bot.PROFILE["llm"]["provider"]
                bot.LLM_MODEL = bot.PROFILE["llm"]["model"]
                bot.update_agentic_tool_profile(enabled=agentic_enabled, auto_reminders=auto_reminders, auto_web_lookup=auto_web_lookup, save=False)
                bot.update_runtime_profile({"preferred_embedding_models": [item.strip() for item in preferred_models.split(",") if item.strip()], "max_thinking_time_seconds": max_thinking_time_seconds, "stream_max_chars": stream_max_chars}, save=False)
                bot.update_cadence_profile(family_echo_turn_interval=family_echo_turn_interval, wisdom_turn_interval=wisdom_turn_interval, life_pattern_queue_limit=life_pattern_queue_limit, save=False)
                bot.update_relationship_calibration_profile(enabled=calibration_enabled, opening_line=opening_line, save=False)
                bot.PROFILE["notifications"] = {
                    "enabled": bool(notify_enabled),
                    "backend": str(notification_backend or "auto").strip().lower() or "auto",
                    "quiet_hours_start": int(quiet_hours_start),
                    "quiet_hours_end": int(quiet_hours_end),
                    "notify_patterns": bool(notify_patterns),
                    "notify_reminders": bool(notify_reminders),
                }
                bot.save_profile()
                st.success("Preferences saved.")
                st.rerun()

