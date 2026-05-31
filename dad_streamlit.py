import asyncio
import base64
import json
import datetime
import json
import os
import traceback
import uuid
from contextlib import suppress
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import streamlit as st
import streamlit.components.v1 as components

from dadbot.components.voice import render_realtime_voice_call, render_reply_tts, render_voice_controls
from dadbot.config import DadBotConfig, DadRuntimeConfig
from dadbot.contracts import DadBotContext
from dadbot.core.dadbot import DadBot
from dadbot.managers.memory_manager import MemoryManager
from dadbot.managers.mood_manager import MoodManager
from dadbot.managers.profile_runtime import ProfileRuntimeManager
from dadbot.managers.relationship_manager import RelationshipManager
from dadbot.managers.tts import TTSManager
from dadbot.runtime.model.model_call_port import ModelConfig
from dadbot.ui.media.service import MediaService
from dadbot_system.events import InMemoryEventBus


def _now_stamp() -> str:
    return datetime.datetime.now().strftime("%H:%M:%S")


def _ensure_ui_runtime_state() -> None:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "chat_threads" not in st.session_state:
        seeded = list(st.session_state.chat_history or [])
        st.session_state.chat_threads = {"default": seeded}

    if "active_thread_id" not in st.session_state:
        st.session_state.active_thread_id = "default"

    if "runtime_events" not in st.session_state:
        st.session_state.runtime_events = []

    if "runtime_guardrails" not in st.session_state:
        st.session_state.runtime_guardrails = []

    if "message_outbox" not in st.session_state:
        st.session_state.message_outbox = []


def _presence_avatar_markup() -> str:
    live_video_candidates = (
        (Path("static/dad_avatar_live.webm"), "video/webm"),
        (Path("static/dad_avatar_live.mp4"), "video/mp4"),
        (Path("static/dad_avatar_live.mov"), "video/quicktime"),
    )
    for file_path, mime in live_video_candidates:
        if file_path.exists():
            with suppress(Exception):
                payload = base64.b64encode(file_path.read_bytes()).decode("ascii")
                return (
                    '<video class="presence-avatar-media" autoplay muted loop playsinline>'
                    f'<source src="data:{mime};base64,{payload}" type="{mime}" />'
                    "</video>"
                )

    live_image_candidates = (
        (Path("static/assets/dad_avatar.jpg"), "image/jpeg"),
        (Path("static/dad_avatar_live.gif"), "image/gif"),
        (Path("static/dad_avatar.gif"), "image/gif"),
        (Path("static/dad_avatar.png"), "image/png"),
        (Path("static/dad_avatar.jpg"), "image/jpeg"),
        (Path("static/dad_avatar.jpeg"), "image/jpeg"),
        (Path("static/dad_avatar.webp"), "image/webp"),
    )
    for file_path, mime in live_image_candidates:
        if file_path.exists():
            with suppress(Exception):
                payload = base64.b64encode(file_path.read_bytes()).decode("ascii")
                return f'<img class="presence-avatar-media" src="data:{mime};base64,{payload}" alt="Dad avatar" />'

    return '<div class="presence-avatar-fallback">D</div>'


def _spatial_avatar_embed_url() -> str:
    env_url = str(os.getenv("DADBOT_SPATIAL_AVATAR_URL", "")).strip()
    if env_url:
        return env_url

    with suppress(Exception):
        secret_url = str(st.secrets.get("DADBOT_SPATIAL_AVATAR_URL", "")).strip()
        if secret_url:
            return secret_url

    return ""


def _spatial_avatar_relay_ws_url() -> str:
    env_url = str(os.getenv("DADBOT_SPATIAL_RELAY_WS_URL", "")).strip()
    if env_url:
        return env_url

    with suppress(Exception):
        secret_url = str(st.secrets.get("DADBOT_SPATIAL_RELAY_WS_URL", "")).strip()
        if secret_url:
            return secret_url

    return "ws://127.0.0.1:8787/v1/spatial/ws"


def _spatial_avatar_image_url() -> str:
    env_url = str(os.getenv("DADBOT_SPATIAL_AVATAR_IMAGE_URL", "")).strip()
    if env_url:
        return env_url

    with suppress(Exception):
        secret_url = str(st.secrets.get("DADBOT_SPATIAL_AVATAR_IMAGE_URL", "")).strip()
        if secret_url:
            return secret_url

    return ""


def _spatial_embed_src(embed_url: str) -> str:
    relay_ws = _spatial_avatar_relay_ws_url()
    if not embed_url or not relay_ws:
        return embed_url

    avatar_image = _spatial_avatar_image_url()
    split = urlsplit(embed_url)
    pairs = [
        (k, v)
        for k, v in parse_qsl(split.query, keep_blank_values=True)
        if k not in {"dadbot_ws", "dadbotWs", "dadbot_avatar_image", "avatarImage"}
    ]
    pairs.append(("dadbot_ws", relay_ws))
    pairs.append(("dadbotWs", relay_ws))
    if avatar_image:
        pairs.append(("dadbot_avatar_image", avatar_image))
        pairs.append(("avatarImage", avatar_image))
    query = urlencode(pairs)
    return urlunsplit((split.scheme, split.netloc, split.path, query, split.fragment))


def _spatial_avatar_inline_html(embed_src: str) -> str:
    html_path = Path("static/spatial_live_mock/index.html")
    html_text = html_path.read_text(encoding="utf-8")
    query = urlsplit(embed_src).query if embed_src else ""
    query_json = json.dumps(query)
    avatar_asset = Path("static/assets/dad_avatar.jpg")
    avatar_data_url = ""
    if avatar_asset.exists():
        avatar_bytes = avatar_asset.read_bytes()
        avatar_data_url = f"data:image/jpeg;base64,{base64.b64encode(avatar_bytes).decode('ascii')}"
    avatar_data_json = json.dumps(avatar_data_url)
    html_text = html_text.replace(
        '    const params = new URLSearchParams(window.location.search);',
        '    const injectedQuery = typeof window.__DADBOT_QUERY__ === "string" ? window.__DADBOT_QUERY__ : "";\n'
        '    const params = new URLSearchParams(injectedQuery || window.location.search || "");',
    )
    html_text = html_text.replace(
        '    const avatarImageUrl = params.get("dadbot_avatar_image") || params.get("avatarImage") || "";',
        '    const injectedAvatarImage = typeof window.__DADBOT_AVATAR_IMAGE__ === "string" ? window.__DADBOT_AVATAR_IMAGE__ : "";\n'
        '    const avatarImageUrl = params.get("dadbot_avatar_image") || params.get("avatarImage") || injectedAvatarImage || "";',
    )
    return (
        f"<script>window.__DADBOT_QUERY__ = {query_json};</script>"
        f"<script>window.__DADBOT_AVATAR_IMAGE__ = {avatar_data_json};</script>"
        f"{html_text}"
    )


def _inject_modern_theme() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Serif:wght@500;600&display=swap');

        :root {
            --dad-bg: #10151f;
            --dad-panel: rgba(13, 21, 33, 0.92);
            --dad-panel-strong: rgba(14, 23, 36, 0.98);
            --dad-border: rgba(175, 197, 231, 0.32);
            --dad-ink: #f4f8ff;
            --dad-muted: #c4d3ea;
            --dad-accent: #8da9d6;
            --dad-accent-2: #9dc4ff;
            --dad-shadow: 0 30px 90px rgba(2, 7, 14, 0.55);
        }

        html, body, [class*="css"] {
            font-family: 'IBM Plex Sans', 'Segoe UI', sans-serif;
        }

        .stApp {
            background:
                radial-gradient(circle at 14% 16%, rgba(90, 110, 255, 0.28), transparent 34%),
                radial-gradient(circle at 84% 10%, rgba(79, 205, 225, 0.22), transparent 30%),
                radial-gradient(circle at 76% 84%, rgba(102, 126, 214, 0.22), transparent 30%),
                radial-gradient(circle at 10% 82%, rgba(79, 98, 193, 0.24), transparent 26%),
                linear-gradient(168deg, #080d15 0%, #0b1320 44%, #111c2e 100%);
            color: var(--dad-ink);
        }

        .stApp p,
        .stApp li,
        .stApp label,
        .stApp div {
            color: var(--dad-ink);
        }

        header[data-testid="stHeader"], #MainMenu, footer {
            visibility: hidden;
            height: 0;
        }

        .block-container {
            max-width: 980px;
            padding-top: 0.8rem;
            padding-bottom: 1.5rem;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(6, 11, 19, 0.86), rgba(10, 16, 27, 0.84));
            border-right: 1px solid rgba(255, 255, 255, 0.06);
            backdrop-filter: blur(14px);
        }

        [data-testid="stSidebar"] * {
            color: #f5f2eb !important;
        }

        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] span {
            color: rgba(245, 242, 235, 0.92) !important;
        }

        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            letter-spacing: -0.03em;
        }

        .dad-app-shell {
            margin: 0.15rem 0 0.8rem;
            padding: 0.45rem 0.2rem 0.2rem;
            border-bottom: 1px solid rgba(196, 210, 238, 0.14);
        }

        .dad-app-shell .eyebrow {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            margin-bottom: 0.3rem;
            padding: 0.24rem 0.7rem;
            border-radius: 999px;
            background: rgba(157, 196, 255, 0.16);
            color: #cfe2ff;
            font-size: 0.74rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.12em;
        }

        .dad-app-shell h1 {
            margin: 0;
            color: var(--dad-ink);
            font-family: 'IBM Plex Serif', Georgia, serif;
            font-size: clamp(1.6rem, 2.2vw, 2.3rem);
            letter-spacing: -0.045em;
            line-height: 1;
        }

        .dad-app-shell p {
            margin: 0.25rem 0 0;
            max-width: 72ch;
            color: var(--dad-muted);
            font-size: 0.92rem;
            line-height: 1.45;
        }

        h1, h2, h3 {
            letter-spacing: -0.04em;
            color: var(--dad-ink);
        }

        [data-testid="stChatMessage"] {
            padding: 0.1rem 0;
            max-width: 100%;
        }

        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
            background: var(--dad-panel);
            border: 1px solid var(--dad-border);
            border-radius: 24px;
            padding: 0.96rem 1.08rem;
            box-shadow: 0 14px 30px rgba(2, 8, 16, 0.42);
            backdrop-filter: blur(18px);
            line-height: 1.62;
            color: var(--dad-ink);
        }

        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p {
            margin-bottom: 0.25rem;
        }

        [data-testid="stChatMessage"] [data-testid="stCaptionContainer"] {
            color: var(--dad-muted);
        }

        div[data-testid="stChatInput"] {
            background: rgba(11, 19, 31, 0.96) !important;
            border: 1px solid rgba(179, 201, 235, 0.34) !important;
            border-radius: 20px !important;
            box-shadow: 0 18px 38px rgba(2, 8, 16, 0.44) !important;
            border-top: 1px solid rgba(161, 179, 206, 0.2);
            padding: 0.6rem 0.7rem 0.55rem;
            margin-top: 0.8rem;
        }

        div[data-testid="stElementContainer"]:has(div[data-testid="stChatInput"]) {
            background: transparent !important;
        }

        div[data-testid="stChatInput"] > div,
        div[data-testid="stChatInput"] [data-baseweb="textarea"],
        div[data-testid="stChatInput"] [data-baseweb="base-input"] {
            background: transparent !important;
        }

        div[data-testid="stChatInput"] * {
            background-color: transparent !important;
        }

        div[data-testid="stChatInput"] textarea,
        div[data-testid="stChatInput"] input {
            border-radius: 999px !important;
            background: rgba(10, 18, 30, 0.98) !important;
            color: #f5f9ff !important;
            border: 1px solid rgba(179, 201, 235, 0.38) !important;
            box-shadow: 0 14px 34px rgba(2, 8, 16, 0.38);
        }

        div[data-testid="stChatInput"] textarea::placeholder,
        div[data-testid="stChatInput"] input::placeholder {
            color: #9db3d6 !important;
            opacity: 1 !important;
        }

        div[data-testid="stChatInput"] button {
            color: #d9e7ff !important;
        }

        div[data-testid="stChatInput"] [data-testid="stChatInputFileUploadButton"] button,
        div[data-testid="stChatInput"] [data-testid="stChatInputSubmitButton"] {
            background: rgba(148, 184, 235, 0.08) !important;
            border: 1px solid rgba(179, 201, 235, 0.24) !important;
        }

        div[data-testid="stButton"] button {
            border-radius: 14px;
            border: 1px solid rgba(39, 75, 95, 0.14);
            background: linear-gradient(180deg, rgba(255,255,255,0.96), rgba(244,240,233,0.92));
            color: var(--dad-ink);
            box-shadow: 0 10px 24px rgba(22, 33, 47, 0.08);
            transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease;
        }

        div[data-testid="stButton"] button:hover {
            transform: translateY(-1px);
            border-color: rgba(39, 75, 95, 0.28);
            box-shadow: 0 14px 28px rgba(22, 33, 47, 0.12);
        }

        div[data-testid="stMetric"] {
            background: var(--dad-panel);
            border: 1px solid var(--dad-border);
            border-radius: 20px;
            padding: 0.85rem 0.9rem;
            box-shadow: 0 12px 28px rgba(22, 33, 47, 0.05);
        }

        div[data-testid="stDataFrame"] {
            border-radius: 18px;
            overflow: hidden;
            border: 1px solid var(--dad-border);
            box-shadow: 0 12px 30px rgba(22, 33, 47, 0.06);
        }

        .chat-layout {
            margin-top: 0.25rem;
        }

        .chat-layout .main-column {
            max-width: 760px;
            margin: 0 auto;
        }

        .chat-shell {
            padding: 0.25rem 0 0;
        }

        .presence-shell {
            margin: 0.1rem 0 1rem;
            padding: 1.1rem 1rem 1rem;
            border-radius: 34px;
            border: 1px solid rgba(174, 198, 235, 0.24);
            background:
                radial-gradient(circle at 50% 10%, rgba(170, 203, 255, 0.2), transparent 45%),
                linear-gradient(180deg, rgba(16, 25, 40, 0.95), rgba(13, 21, 33, 0.93));
            box-shadow: 0 24px 58px rgba(1, 7, 15, 0.52);
            text-align: center;
        }

        .presence-pill {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            padding: 0.3rem 0.82rem;
            border-radius: 999px;
            border: 1px solid rgba(174, 198, 235, 0.35);
            color: #d7e6ff;
            background: rgba(157, 196, 255, 0.12);
            letter-spacing: 0.12em;
            text-transform: uppercase;
            font-size: 0.72rem;
            font-weight: 700;
        }

        .presence-avatar-wrap {
            position: relative;
            width: 212px;
            height: 212px;
            border-radius: 999px;
            margin: 0.9rem auto 0;
            padding: 8px;
            background: radial-gradient(circle at 50% 50%, rgba(164, 205, 255, 0.46), rgba(164, 205, 255, 0.12) 58%, rgba(164, 205, 255, 0.04) 72%);
            box-shadow: 0 24px 42px rgba(2, 10, 20, 0.55);
        }

        .presence-embed-wrap {
            position: relative;
            width: min(720px, 100%);
            height: 380px;
            border-radius: 24px;
            margin: 0.9rem auto 0;
            overflow: hidden;
            border: 1px solid rgba(174, 198, 235, 0.26);
            background: rgba(9, 15, 25, 0.9);
            box-shadow: 0 24px 42px rgba(2, 10, 20, 0.55);
        }

        .presence-avatar-wrap img,
        .presence-avatar-wrap video,
        .presence-avatar-wrap .presence-avatar-media,
        .presence-avatar-fallback {
            width: 100%;
            height: 100%;
            border-radius: 999px;
            object-fit: cover;
            display: block;
        }

        .presence-avatar-fallback {
            background: linear-gradient(180deg, #31466f, #1c2942);
            color: #edf5ff;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 3.2rem;
            font-weight: 600;
            letter-spacing: -0.04em;
        }

        .chat-hero .hero-copy {
            margin-top: 0.45rem;
            color: var(--dad-muted);
            font-size: 0.9rem;
            line-height: 1.58;
            max-width: 62ch;
        }

        .chat-hero .hero-badges {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-top: 0.85rem;
        }

        .chat-hero .hero-badge {
            padding: 0.38rem 0.72rem;
            border-radius: 999px;
            background: rgba(255,255,255,0.82);
            border: 1px solid rgba(53, 64, 78, 0.08);
            color: var(--dad-ink);
            font-size: 0.8rem;
            font-weight: 600;
        }

        @media (max-width: 960px) {
            .presence-avatar-wrap {
                width: 172px;
                height: 172px;
            }

            .presence-embed-wrap {
                height: 320px;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_app_shell(page: str) -> None:
    page_meta = {
        "Chat": ("Conversation surface", "Warm, direct, live chat with profile-safe replies and durable thread handling."),
        "Smart Home": ("Home control", "MQTT controls with a cleaner operational panel."),
        "Voice": ("Voice runtime", "Push-to-talk and realtime voice surfaces with the same dad persona."),
        "Status": ("System status", "Operational health, runtime events, and guardrail feed in one place."),
        "Workshop": ("Workbench", "Recovery tools, outbox management, and thread export/import."),
    }
    eyebrow, description = page_meta.get(page, ("DadBot", "A focused control surface for the family runtime."))
    st.markdown(
        f"""
        <section class="dad-app-shell">
          <div class="eyebrow">{eyebrow}</div>
          <h1>DadBot</h1>
          <p>{description}</p>
          <div class="dad-chip-row">
            <span class="dad-chip">Persona-safe replies</span>
            <span class="dad-chip">Threaded chat</span>
            <span class="dad-chip">Live runtime</span>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def _active_thread_id() -> str:
    return str(st.session_state.get("active_thread_id") or "default")


def _thread_ids() -> list[str]:
    return sorted(str(k) for k in dict(st.session_state.get("chat_threads") or {}).keys())


def _thread_messages(thread_id: str | None = None) -> list[dict[str, Any]]:
    tid = str(thread_id or _active_thread_id())
    threads = dict(st.session_state.get("chat_threads") or {})
    values = list(threads.get(tid) or [])
    return [dict(item) for item in values if isinstance(item, dict)]


def _chat_input_payload(value: Any) -> tuple[str, list[Any]]:
    if value is None:
        return "", []
    if isinstance(value, str):
        return value, []

    text = str(getattr(value, "text", "") or "")
    files = getattr(value, "files", None)

    if isinstance(value, dict):
        text = str(value.get("text") or text or "")
        if files is None:
            files = value.get("files")

    return text, list(files or [])


def _set_thread_messages(messages: list[dict[str, Any]], thread_id: str | None = None) -> None:
    tid = str(thread_id or _active_thread_id())
    threads = dict(st.session_state.get("chat_threads") or {})
    threads[tid] = [dict(item) for item in list(messages or []) if isinstance(item, dict)]
    st.session_state.chat_threads = threads
    if tid == _active_thread_id():
        st.session_state.chat_history = list(threads[tid])


def _append_chat_message(
    role: str,
    text: str,
    *,
    stamp: str,
    thread_id: str | None = None,
    audio_base64: str = "",
    audio_mime: str = "",
) -> None:
    tid = str(thread_id or _active_thread_id())
    messages = _thread_messages(tid)
    message = {
        "role": str(role or "assistant"),
        "text": str(text or ""),
        "time": str(stamp or _now_stamp()),
    }
    if audio_base64:
        message["audio_base64"] = str(audio_base64)
        message["audio_mime"] = str(audio_mime or "audio/mpeg")
    messages.append(message)
    _set_thread_messages(messages, tid)


def _switch_active_thread(thread_id: str) -> None:
    tid = str(thread_id or "default")
    threads = dict(st.session_state.get("chat_threads") or {})
    if tid not in threads:
        threads[tid] = []
        st.session_state.chat_threads = threads
    st.session_state.active_thread_id = tid
    st.session_state.chat_history = _thread_messages(tid)


def _create_new_thread() -> str:
    tid = f"thread-{uuid.uuid4().hex[:8]}"
    threads = dict(st.session_state.get("chat_threads") or {})
    threads[tid] = []
    st.session_state.chat_threads = threads
    _switch_active_thread(tid)
    return tid


def _fork_active_thread() -> str:
    source_id = _active_thread_id()
    source = _thread_messages(source_id)
    tid = f"fork-{uuid.uuid4().hex[:8]}"
    threads = dict(st.session_state.get("chat_threads") or {})
    threads[tid] = list(source)
    st.session_state.chat_threads = threads
    _switch_active_thread(tid)
    return tid


def _record_runtime_event(event_type: str, summary: str, *, severity: str = "info", payload: dict[str, Any] | None = None) -> None:
    events = list(st.session_state.get("runtime_events") or [])
    events.append(
        {
            "time": _now_stamp(),
            "thread_id": _active_thread_id(),
            "event_type": str(event_type or "event"),
            "severity": str(severity or "info"),
            "summary": str(summary or ""),
            "payload": dict(payload or {}),
        }
    )
    st.session_state.runtime_events = events[-250:]


def _speak_reply_client_side(text: str) -> None:
    content = str(text or "").strip()
    if not content:
        return
    payload = json.dumps(content)
    components.html(
        f"""
        <script>
        (function() {{
            const text = {payload};
            if (!text || !('speechSynthesis' in window)) return;
            try {{
                const synth = window.speechSynthesis;
                const voices = synth.getVoices() || [];
                const prefer = (v) => /en-US/i.test(v.lang || '') && /(neural|guy|davis|christopher|eric|david|male)/i.test((v.name || '') + ' ' + (v.voiceURI || ''));
                const voice = voices.find(prefer) || voices.find((v) => /en-US/i.test(v.lang || '')) || voices[0] || null;
                const utter = new SpeechSynthesisUtterance(text);
                if (voice) utter.voice = voice;
                utter.rate = 1.0;
                utter.pitch = 1.0;
                utter.volume = 1.0;
                synth.cancel();
                synth.speak(utter);
            }} catch (_err) {{}}
        }})();
        </script>
        """,
        height=0,
    )


def _install_client_speech_cancel_guard() -> None:
        components.html(
                """
                <script>
                (function() {
                    if (window.__dadbotSpeechGuardInstalled) return;
                    window.__dadbotSpeechGuardInstalled = true;
                    const stopSpeech = function () {
                        if (!('speechSynthesis' in window)) return;
                        try { window.speechSynthesis.cancel(); } catch (_err) {}
                    };
                    stopSpeech();
                    window.addEventListener('beforeunload', stopSpeech);
                    window.addEventListener('pagehide', stopSpeech);
                    document.addEventListener('visibilitychange', function () {
                        if (document.visibilityState === 'hidden') stopSpeech();
                    });
                })();
                </script>
                """,
                height=0,
        )


def _record_guardrail(rule: str, detail: str, *, severity: str = "warning") -> None:
    guardrails = list(st.session_state.get("runtime_guardrails") or [])
    guardrails.append(
        {
            "time": _now_stamp(),
            "thread_id": _active_thread_id(),
            "rule": str(rule or "runtime_guardrail"),
            "severity": str(severity or "warning"),
            "detail": str(detail or ""),
        }
    )
    st.session_state.runtime_guardrails = guardrails[-120:]


def _auto_speak_replies_enabled() -> bool:
    if "auto_speak_replies" not in st.session_state:
        st.session_state.auto_speak_replies = True
    return bool(st.session_state.auto_speak_replies)


def _queue_failed_message(prompt: str, error: str) -> None:
    queue = list(st.session_state.get("message_outbox") or [])
    queue.append(
        {
            "thread_id": _active_thread_id(),
            "prompt": str(prompt or ""),
            "error": str(error or ""),
            "time": _now_stamp(),
        }
    )
    st.session_state.message_outbox = queue[-100:]


def _render_thread_sidebar(page: str | None = None) -> None:
    chat_mode = str(page or "").strip().lower() == "chat"
    if chat_mode:
        st.sidebar.markdown(
            "<div style='font-size:0.8rem; letter-spacing:0.14em; text-transform:uppercase; opacity:0.72; margin-bottom:0.35rem;'>Conversation drawer</div>",
            unsafe_allow_html=True,
        )
    else:
        st.sidebar.subheader("Threads")

    thread_ids = _thread_ids()
    active = _active_thread_id()
    if active not in thread_ids:
        _switch_active_thread("default")
        thread_ids = _thread_ids()
        active = _active_thread_id()

    index = thread_ids.index(active) if active in thread_ids else 0
    selected = st.sidebar.selectbox("Active conversation", thread_ids, index=index, key="thread-picker")
    if str(selected) != active:
        _switch_active_thread(str(selected))
        st.rerun()

    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("New chat" if chat_mode else "New", use_container_width=True):
            new_tid = _create_new_thread()
            _record_runtime_event("thread_created", f"Created {new_tid}")
            st.rerun()
    with col2:
        if st.button("Duplicate" if chat_mode else "Fork", use_container_width=True):
            new_tid = _fork_active_thread()
            _record_runtime_event("thread_forked", f"Forked into {new_tid}")
            st.rerun()

    if not chat_mode:
        st.sidebar.caption(f"Messages: {len(_thread_messages(active))}")


def _render_guardrail_strip() -> None:
    guardrails = list(st.session_state.get("runtime_guardrails") or [])
    if not guardrails:
        return
    recent = guardrails[-1]
    sev = str(recent.get("severity") or "warning").lower()
    msg = f"{recent.get('time', '')} · {recent.get('rule', 'guardrail')}: {recent.get('detail', '')}"
    if sev in {"critical", "error"}:
        st.error(msg)
    else:
        st.warning(msg)


def _render_runtime_timeline(limit: int = 30) -> None:
    events = list(st.session_state.get("runtime_events") or [])[-max(1, int(limit)) :]
    if not events:
        st.info("No runtime events yet.")
        return
    rows = []
    for item in reversed(events):
        rows.append(
            {
                "time": str(item.get("time") or ""),
                "thread": str(item.get("thread_id") or "default"),
                "event": str(item.get("event_type") or "event"),
                "severity": str(item.get("severity") or "info"),
                "summary": str(item.get("summary") or ""),
            }
        )
    st.dataframe(rows, use_container_width=True)


def _extract_ollama_text(response: Any) -> str:
    payload: dict[str, Any]
    if hasattr(response, "model_dump"):
        payload = dict(response.model_dump() or {})
    elif isinstance(response, dict):
        payload = dict(response)
    else:
        payload = {}

    message = payload.get("message") if isinstance(payload, dict) else None
    if isinstance(message, dict):
        text = str(message.get("content") or "").strip()
        if text:
            return text

    choices = payload.get("choices") if isinstance(payload, dict) else None
    if isinstance(choices, list) and choices:
        choice = choices[0]
        if isinstance(choice, dict):
            msg = choice.get("message") or choice.get("delta") or {}
            if isinstance(msg, dict):
                text = str(msg.get("content") or "").strip()
                if text:
                    return text

    text = str(payload.get("response") or "").strip() if isinstance(payload, dict) else ""
    return text


class _StreamlitControlPlane:
    def __init__(self, service: "StrictLLMService") -> None:
        self._service = service

    async def execute_from_graph_context(self, turn_context: Any, rich_context: dict[str, Any] | None = None):
        reply = await self._service.run_agent(turn_context, dict(rich_context or {}))
        return str(reply or ""), False


class StrictLLMService:
    def __init__(self, model_name: str, bot: Any = None) -> None:
        self.model_name = str(model_name or "llama3.2:latest")
        self.bot = bot
        self.control_plane = _StreamlitControlPlane(self)

    @staticmethod
    def _collect_attachments(context: Any, rich_context: dict[str, Any]) -> list[dict[str, Any]]:
        def _as_items(value: Any) -> list[Any]:
            if isinstance(value, list):
                return list(value)
            if isinstance(value, tuple):
                return list(value)
            return []

        candidates: list[Any] = []
        if isinstance(rich_context, dict):
            candidates.extend(_as_items(rich_context.get("attachments")))
            candidates.extend(_as_items(rich_context.get("norm_attachments")))

        state = getattr(context, "state", None)
        if isinstance(state, dict):
            candidates.extend(_as_items(state.get("attachments")))
            candidates.extend(_as_items(state.get("norm_attachments")))

        candidates.extend(_as_items(getattr(context, "attachments", [])))

        attachments: list[dict[str, Any]] = []
        for attachment in candidates:
            if not isinstance(attachment, dict):
                continue
            kind = str(attachment.get("type") or "").strip().lower()
            if kind not in {"image", "audio", "document"}:
                continue
            attachments.append(dict(attachment))
        return attachments[:6]

    @staticmethod
    def _extract_image_payload(attachments: list[dict[str, Any]]) -> list[str]:
        images: list[str] = []
        for attachment in list(attachments or []):
            if not isinstance(attachment, dict):
                continue
            kind = str(attachment.get("type") or "").strip().lower()
            if kind != "image":
                continue
            image_b64 = str(attachment.get("image_b64") or attachment.get("data_b64") or "").strip()
            if image_b64:
                images.append(image_b64)

        deduped: list[str] = []
        seen: set[str] = set()
        for payload in images:
            if payload in seen:
                continue
            seen.add(payload)
            deduped.append(payload)
        return deduped[:6]

    @staticmethod
    def _model_supports_image_input(model_name: str) -> bool:
        lowered = str(model_name or "").strip().lower()
        if not lowered:
            return False
        hints = (
            "vision",
            "llava",
            "bakllava",
            "moondream",
            "minicpm-v",
            "qwen2-vl",
            "qwen2.5-vl",
            "qwen2.5vl",
            "gemma3",
        )
        return any(hint in lowered for hint in hints)

    @staticmethod
    def _augment_messages_with_attachment_note(
        messages: list[dict[str, Any]],
        attachments: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        image_names = [str(item.get("name") or "image") for item in attachments if item.get("type") == "image"]
        if not image_names:
            return messages

        note = "User uploaded image(s): " + ", ".join(image_names[:6])
        enriched = [dict(item) for item in list(messages or [])]
        for idx in range(len(enriched) - 1, -1, -1):
            if str(enriched[idx].get("role") or "").lower() != "user":
                continue
            content = str(enriched[idx].get("content") or "").strip()
            enriched[idx]["content"] = f"{content}\n\n{note}".strip()
            return enriched

        enriched.append({"role": "user", "content": note})
        return enriched

    def _to_messages(
        self,
        context: Any,
        rich_context: dict[str, Any],
        attachments: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        def _resolve_current_mood() -> str:
            mood_candidates: list[str] = []
            if isinstance(rich_context, dict):
                for key in ("mood", "current_mood"):
                    value = rich_context.get(key)
                    if value:
                        mood_candidates.append(str(value))
                temporal = rich_context.get("temporal")
                if isinstance(temporal, dict) and temporal.get("mood"):
                    mood_candidates.append(str(temporal.get("mood")))

            state = getattr(context, "state", None)
            if isinstance(state, dict):
                for key in ("mood", "current_mood"):
                    value = state.get(key)
                    if value:
                        mood_candidates.append(str(value))
                temporal = state.get("temporal")
                if isinstance(temporal, dict) and temporal.get("mood"):
                    mood_candidates.append(str(temporal.get("mood")))

            last_saved_mood = getattr(self.bot, "last_saved_mood", None)
            if callable(last_saved_mood):
                with suppress(Exception):
                    value = last_saved_mood()
                    if value:
                        mood_candidates.append(str(value))

            normalize_mood = getattr(self.bot, "normalize_mood", None)
            for candidate in mood_candidates:
                normalized = str(candidate or "").strip()
                if not normalized:
                    continue
                if callable(normalize_mood):
                    with suppress(Exception):
                        return str(normalize_mood(normalized) or "neutral")
                return normalized.lower()
            return "neutral"

        def _build_persona_system_prompt(
            user_input: str,
            current_mood: str,
            selected_attachments: list[dict[str, Any]],
        ) -> str:
            if self.bot is None:
                return ""

            prompt_assembly = getattr(self.bot, "prompt_assembly", None)
            request_builder = getattr(prompt_assembly, "build_request_system_prompt", None)
            if callable(request_builder):
                with suppress(Exception):
                    return str(
                        request_builder(
                            user_input,
                            current_mood,
                            selected_attachments,
                        )
                        or ""
                    ).strip()

            sections: list[str] = []
            core_persona = getattr(self.bot, "build_core_persona_prompt", None)
            if callable(core_persona):
                with suppress(Exception):
                    content = str(core_persona() or "").strip()
                    if content:
                        sections.append(content)

            tone_context = getattr(self.bot, "tone_context", None)
            mood_builder = getattr(tone_context, "build_mood_context", None)
            if callable(mood_builder):
                with suppress(Exception):
                    content = str(mood_builder(current_mood) or "").strip()
                    if content:
                        sections.append(content)

            return "\n\n".join(section for section in sections if section).strip()

        raw_messages = rich_context.get("messages") if isinstance(rich_context, dict) else None
        normalized: list[dict[str, Any]] = []
        if isinstance(raw_messages, list):
            for item in raw_messages:
                if not isinstance(item, dict):
                    continue
                role = str(item.get("role") or "user").strip().lower() or "user"
                content = str(item.get("content") or "").strip()
                if content:
                    normalized.append({"role": role, "content": content})

        if not normalized:
            user_input = str(getattr(context, "user_input", "") or "").strip()
            if user_input:
                normalized = [{"role": "user", "content": user_input}]
            else:
                normalized = [{"role": "user", "content": "Hello"}]

        selected_attachments = list(attachments or self._collect_attachments(context, rich_context))

        has_system_message = any(
            str(message.get("role") or "").strip().lower() == "system"
            for message in normalized
            if isinstance(message, dict)
        )
        if not has_system_message:
            user_input = ""
            for message in reversed(normalized):
                if str(message.get("role") or "").strip().lower() != "user":
                    continue
                user_input = str(message.get("content") or "").strip()
                if user_input:
                    break

            current_mood = _resolve_current_mood()
            system_prompt = _build_persona_system_prompt(
                user_input=user_input,
                current_mood=current_mood,
                selected_attachments=selected_attachments,
            )
            if system_prompt:
                normalized = [{"role": "system", "content": system_prompt}, *normalized]

        images = self._extract_image_payload(selected_attachments)
        if images:
            attached = False
            for idx in range(len(normalized) - 1, -1, -1):
                if str(normalized[idx].get("role") or "").lower() == "user":
                    enriched = dict(normalized[idx])
                    enriched["images"] = images
                    normalized[idx] = enriched
                    attached = True
                    break
            if not attached:
                normalized.append(
                    {
                        "role": "user",
                        "content": "Please review the attached image(s).",
                        "images": images,
                    }
                )

        return normalized

    def _chat_sync(self, messages: list[dict[str, Any]], attachments: list[dict[str, Any]]) -> str:
        import ollama

        selected_model = self.model_name
        image_turn = any(bool(list(item.get("images") or [])) for item in list(messages or []))

        if image_turn and not self._model_supports_image_input(selected_model):
            finder = getattr(self.bot, "find_available_vision_model", None)
            if callable(finder):
                with suppress(Exception):
                    candidate = str(finder() or "").strip()
                    if candidate:
                        selected_model = candidate

        selected_messages = list(messages or [])
        if image_turn and not self._model_supports_image_input(selected_model):
            selected_messages = self._augment_messages_with_attachment_note(selected_messages, attachments)

        response = ollama.chat(model=selected_model, messages=selected_messages)
        text = _extract_ollama_text(response)
        if text:
            return text
        raise RuntimeError("Ollama returned an empty response")

    async def run_agent(self, context: Any, rich_context: dict[str, Any]) -> str:
        attachments = self._collect_attachments(context, rich_context)
        messages = self._to_messages(context, rich_context, attachments)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self._chat_sync(messages, attachments))


def _build_bot() -> DadBot:
    class MinimalPersistenceService:
        def __init__(self) -> None:
            self.bot: Any | None = None

        def finalize_turn(self, turn_context: Any, result: Any):
            if isinstance(result, tuple):
                raw_reply = result[0] if len(result) >= 1 else ""
                should_end = bool(result[1]) if len(result) >= 2 else False
            else:
                raw_reply = result
                should_end = bool(getattr(turn_context, "state", {}).get("should_end", False))

            reply = str(raw_reply or "")
            bot = self.bot
            if bot is not None:
                finalization = getattr(bot, "reply_finalization", None)
                finalize = getattr(finalization, "finalize", None)
                if callable(finalize):
                    mood = str(getattr(turn_context, "state", {}).get("mood") or "neutral")
                    user_input = str(getattr(turn_context, "user_input", "") or "")
                    with suppress(Exception):
                        reply = str(finalize(reply, mood, user_input=user_input) or reply)
                else:
                    append_signoff = getattr(finalization, "append_signoff", None)
                    if callable(append_signoff):
                        with suppress(Exception):
                            reply = str(append_signoff(reply) or reply)

            return reply, should_end

        def save_turn(self, turn_context: Any, result: Any) -> None:
            pass

        def save_turn_event(self, event: dict[str, Any]) -> None:
            pass

        def save_graph_checkpoint(self, checkpoint: dict[str, Any], _skip_turn_event: bool = False) -> None:
            pass

    class MinimalSafetyService:
        def tick(self, context: Any = None) -> dict[str, Any]:
            return {"status": "ok"}

    class MinimalContextService:
        def build_context(self, turn_context: Any) -> dict[str, Any]:
            return {"messages": [], "temporal": dict(getattr(turn_context, "state", {}).get("temporal") or {})}

    class MinimalMaintenanceService:
        def tick(self, context: Any = None) -> dict[str, Any]:
            return {"status": "ok"}

    class MinimalReflectionService:
        def reflect_after_turn(self, turn_text: str, current_mood: str, reply_text: str) -> dict[str, Any]:
            return {"reflection": "noop"}

        def reflect(self, turn_context: Any, result: Any) -> dict[str, Any]:
            return {"reflection": "noop"}

    class ModelRuntimeStub:
        def __init__(self, active_model: str):
            self._active_model = str(active_model or "llama3.2:latest")

        def initialize_tokenizer(self, model_name: str | None = None):
            return None

        def current_tokenizer(self, model_name: str | None = None):
            class DummyTokenizer:
                @staticmethod
                def encode(text: str):
                    return str(text or "").split()

            return DummyTokenizer()

        @staticmethod
        def model_chars_per_token_estimate(model_name: str | None = None) -> float:
            return 4.0

        @staticmethod
        def normalized_llm_provider() -> str:
            return "ollama"

        def normalized_llm_model(self, model_name: str | None = None) -> str:
            return str(model_name or self._active_model)

        @staticmethod
        def resolve_temperature(options: dict[str, Any] | None = None, default: float = 0.7) -> float:
            if isinstance(options, dict):
                with suppress(TypeError, ValueError):
                    return float(options.get("temperature", default))
            return float(default)

        def litellm_model_identifier(self, model_name: str | None = None) -> str:
            return self.normalized_llm_model(model_name)

        @staticmethod
        def extract_stream_chunk_content(chunk: Any) -> str:
            if isinstance(chunk, dict):
                choices = chunk.get("choices")
                if isinstance(choices, list) and choices:
                    msg = choices[0].get("delta") or choices[0].get("message") or {}
                    if isinstance(msg, dict):
                        return str(msg.get("content") or "")
            return ""

        def ollama_status(self) -> dict[str, Any]:
            try:
                import ollama

                models = ollama.list()
                payload = models.model_dump() if hasattr(models, "model_dump") else dict(models or {})
                items = list(payload.get("models") or []) if isinstance(payload, dict) else []
                return {
                    "connected": True,
                    "model": self._active_model,
                    "models_count": len(items),
                }
            except Exception as exc:
                return {
                    "connected": False,
                    "model": self._active_model,
                    "error": str(exc),
                }

    class _TempDadBot:
        pass

    temp_context = DadBotContext(_TempDadBot())
    memory_manager = MemoryManager(temp_context)
    relationship_manager = RelationshipManager(temp_context)
    mood_manager = MoodManager(temp_context)
    profile_runtime = ProfileRuntimeManager(temp_context)

    runtime_config = DadRuntimeConfig()
    config = DadBotConfig(runtime_config=runtime_config)
    event_bus = InMemoryEventBus()
    model_name = "llama3.2:latest"
    model_config = ModelConfig(active_model=model_name)
    model_runtime_stub = ModelRuntimeStub(model_config.active_model)

    from dadbot.core.interfaces import HealthService, InferenceService
    from dadbot.core.services import build_services
    from dadbot.registry import ServiceRegistry

    class StrictHealthService(HealthService):
        def tick(self, context: Any = None):
            return {"status": "ok"}

    class StrictInferenceService(StrictLLMService, InferenceService):
        pass

    registry = ServiceRegistry()
    for key, value in build_services().items():
        registry.register(key, value)

    llm_service = StrictInferenceService(model_name)
    registry.register("llm", llm_service)
    registry.register("agent_service", llm_service)
    registry.register("health", StrictHealthService())
    registry.register("context_service", MinimalContextService())
    registry.register("maintenance_service", MinimalMaintenanceService())
    persistence_service = MinimalPersistenceService()
    registry.register("persistence_service", persistence_service)
    registry.register("safety_service", MinimalSafetyService())
    registry.register("reflection", MinimalReflectionService())

    class PatchedDadBot(DadBot):
        def _get_turn_orchestrator(self):
            existing = getattr(self, "_turn_orchestrator", None)
            if existing is not None:
                return existing
            from dadbot.core.orchestrator import DadBotOrchestrator

            orchestrator = DadBotOrchestrator(
                registry=registry,
                config_path="config.yaml",
                bot=self,
                strict=True,
            )
            self._turn_orchestrator = orchestrator
            return orchestrator

    bot = PatchedDadBot(
        config=config,
        runtime_config=runtime_config,
        memory_manager=None,
        relationship_manager=None,
        mood_manager=None,
        profile_runtime=None,
        event_bus=event_bus,
        model_runtime=model_runtime_stub,
        validate_managers=False,
    )

    bot._memory_manager = memory_manager
    bot._relationship_manager = relationship_manager
    bot._mood_manager = mood_manager
    bot._profile_runtime = profile_runtime
    bot.services = None
    llm_service.bot = bot
    persistence_service.bot = bot

    with suppress(Exception):
        profile_path = getattr(runtime_config, "profile_path", None)
        if profile_path and Path(profile_path).is_file():
            loaded_profile = json.loads(Path(profile_path).read_text(encoding="utf-8"))
            if isinstance(loaded_profile, dict):
                bot.PROFILE = loaded_profile
    if not isinstance(getattr(bot, "PROFILE", None), dict):
        bot.PROFILE = {}
    bot.PROFILE.setdefault("voice", {})

    return bot


_STREAMLIT_RUNTIME_BUILD = "2026-05-31-persona-system-prompt-v10"

st.set_page_config(
    page_title="DadBot",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="collapsed",
)

if "bot" not in st.session_state:
    st.session_state.bot = _build_bot()
    st.session_state._streamlit_runtime_build = _STREAMLIT_RUNTIME_BUILD
elif str(st.session_state.get("_streamlit_runtime_build") or "") != _STREAMLIT_RUNTIME_BUILD:
    st.session_state.bot = _build_bot()
    st.session_state._streamlit_runtime_build = _STREAMLIT_RUNTIME_BUILD

_ensure_ui_runtime_state()
_switch_active_thread(_active_thread_id())
_inject_modern_theme()

bot = st.session_state.bot
media_service = MediaService()


def _fallback_dad_story() -> str:
    family = getattr(bot, "FAMILY", {}) if bot is not None else {}
    education = getattr(bot, "EDUCATION", {}) if bot is not None else {}
    dad = family.get("dad", {}) if isinstance(family, dict) else {}
    style = getattr(getattr(bot, "profile_runtime", None), "style", {}) or {}
    listener_name = str(style.get("listener_name") or "Tony")

    birthplace = str(dad.get("birthplace") or "Providence, Rhode Island")
    moved_to = str(dad.get("moved_to") or "Albion, Maine")
    move_age = int(dad.get("move_age") or 5)
    boarding_school = str(education.get("boarding_school") or "Kents Hill School")
    university = str(education.get("university") or "University of Maine")

    return (
        f"I'm your dad - born in {birthplace}. "
        f"We moved to {moved_to} when I was {move_age}. "
        f"I went to {boarding_school} and then {university} for English. "
        f"You're my son, {listener_name}."
    )


def _fallback_dad_full_story() -> str:
    profile = getattr(bot, "PROFILE", {}) if bot is not None else {}
    family = getattr(bot, "FAMILY", {}) if bot is not None else {}
    education = getattr(bot, "EDUCATION", {}) if bot is not None else {}
    style = getattr(getattr(bot, "profile_runtime", None), "style", {}) or {}
    sports = profile.get("sports_preferences", {}) if isinstance(profile, dict) else {}

    listener_name = str(style.get("listener_name") or "Tony")
    dad = family.get("dad", {}) if isinstance(family, dict) else {}
    birthplace = str(dad.get("birthplace") or "Providence, Rhode Island")
    moved_to = str(dad.get("moved_to") or "Albion, Maine")
    move_age = int(dad.get("move_age") or 5)
    childhood = str(dad.get("childhood") or "grew up in a small town in Maine")
    boarding_school = str(education.get("boarding_school") or "Kents Hill School")
    university = str(education.get("university") or "University of Maine")
    degree = str(education.get("degree") or "English")
    baseball_team = str(sports.get("baseball_team") or "New York Yankees")
    football_team = str(sports.get("football_team") or "Chicago Bears")
    soccer_team = str(sports.get("soccer_team") or "Naples (Napoli)")
    basketball_interest = str(sports.get("basketball_interest") or "don't really watch basketball")

    return (
        f"Full version, {listener_name}: I was born in {birthplace}, and we moved to {moved_to} when I was {move_age}. "
        f"I {childhood}. I went to {boarding_school}, then {university}, where I studied {degree}. "
        f"I care a lot about family continuity and keeping life steady over time. "
        f"For sports, I'm a {baseball_team} fan in baseball, a {football_team} fan in football, and a {soccer_team} fan in soccer, and I {basketball_interest}."
    )


def _fallback_dad_age() -> str:
    style = getattr(getattr(bot, "profile_runtime", None), "style", {}) or {}
    listener_name = str(style.get("listener_name") or "Tony")
    age_on_date = getattr(bot, "age_on_date", None)
    dad_birthdate = getattr(bot, "DAD_BIRTHDATE", None)
    dad_age = 47
    if callable(age_on_date) and dad_birthdate is not None:
        with suppress(Exception):
            dad_age = int(age_on_date(dad_birthdate))
    return f"I'm {dad_age} years old, {listener_name}."


def _fallback_preference_reply(user_input: str = "") -> str:
    prompt = str(user_input or "").strip().lower()
    style = getattr(getattr(bot, "profile_runtime", None), "style", {}) or {}
    listener_name = str(style.get("listener_name") or "Tony")
    profile = getattr(bot, "PROFILE", {}) if bot is not None else {}
    sports = profile.get("sports_preferences", {}) if isinstance(profile, dict) else {}
    baseball_team = str(sports.get("baseball_team") or "New York Yankees")
    football_team = str(sports.get("football_team") or "Chicago Bears")
    soccer_team = str(sports.get("soccer_team") or "Naples (Napoli)")
    basketball_interest = str(sports.get("basketball_interest") or "don't really watch basketball")

    if "baseball" in prompt and "team" in prompt:
        return f"I'm a {baseball_team} fan for baseball, buddy."
    if "football" in prompt and "team" in prompt:
        return f"I'm a {football_team} fan for football."
    if "soccer" in prompt and "team" in prompt:
        return f"For soccer, I'm a {soccer_team} fan."
    if "basketball" in prompt:
        return f"I {basketball_interest}, buddy."
    return (
        f"I keep it simple, {listener_name}: {baseball_team} for baseball, {football_team} for football, "
        f"{soccer_team} for soccer, and I {basketball_interest}."
    )


def _local_prompt_fallback(user_input: str = "") -> str:
    user_text = str(user_input or "").strip().lower()
    if not user_text:
        return ""

    identity_prompt_markers = (
        "who are you",
        "what's your story",
        "whats your story",
        "tell me your story",
        "your story",
        "about yourself",
        "tell me about yourself",
    )
    detail_markers = (
        "full life story",
        "full story",
        "detailed",
        "detail",
        "long version",
        "more about you",
        "what can you tell me about you",
    )
    if any(marker in user_text for marker in identity_prompt_markers):
        if any(marker in user_text for marker in detail_markers):
            return _fallback_dad_full_story()
        return _fallback_dad_story()

    age_prompt_markers = (
        "how old are you",
        "your age",
        "dad age",
        "what year were you born",
        "when were you born",
        "your birthday",
    )
    if any(marker in user_text for marker in age_prompt_markers):
        return _fallback_dad_age()

    preference_prompt_markers = (
        "favorite",
        "favourite",
        "which team do you like",
        "what team do you like",
        "your team",
        "preference",
        "sports",
        "play any",
        "what can you tell me about you",
        "about you",
    )
    if any(marker in user_text for marker in preference_prompt_markers):
        return _fallback_preference_reply(user_input)

    return ""


def _normalize_ui_reply(reply: Any, *, user_input: str = "") -> str:
    text = str(reply or "")
    finalization = getattr(bot, "reply_finalization", None)
    user_text = str(user_input or "").strip().lower()

    # Deterministic guard: profile fact routing for identity/story prompts.
    identity_prompt_markers = (
        "who are you",
        "what's your story",
        "whats your story",
        "tell me your story",
        "your story",
        "about yourself",
        "tell me about yourself",
    )
    detail_markers = (
        "full life story",
        "full story",
        "detailed",
        "detail",
        "long version",
        "more about you",
        "what can you tell me about you",
    )
    if any(marker in user_text for marker in identity_prompt_markers):
        resolved_fact = ""
        get_fact_reply = getattr(bot, "get_fact_reply", None)
        if callable(get_fact_reply):
            with suppress(Exception):
                fact_reply = str(get_fact_reply(user_input) or "").strip()
                if fact_reply:
                    resolved_fact = fact_reply
        if any(marker in user_text for marker in detail_markers):
            text = _fallback_dad_full_story()
        else:
            text = resolved_fact or _fallback_dad_story()

    age_prompt_markers = (
        "how old are you",
        "your age",
        "dad age",
        "what year were you born",
        "when were you born",
        "your birthday",
    )
    if any(marker in user_text for marker in age_prompt_markers):
        resolved_fact = ""
        get_fact_reply = getattr(bot, "get_fact_reply", None)
        if callable(get_fact_reply):
            with suppress(Exception):
                fact_reply = str(get_fact_reply(user_input) or "").strip()
                if fact_reply:
                    resolved_fact = fact_reply
        text = resolved_fact or _fallback_dad_age()

    preference_prompt_markers = (
        "favorite",
        "favourite",
        "which team do you like",
        "what team do you like",
        "your team",
        "preference",
        "sports",
        "play any",
        "what can you tell me about you",
        "about you",
    )
    if any(marker in user_text for marker in preference_prompt_markers):
        resolved_fact = ""
        get_fact_reply = getattr(bot, "get_fact_reply", None)
        if callable(get_fact_reply):
            with suppress(Exception):
                fact_reply = str(get_fact_reply(user_input) or "").strip()
                if fact_reply:
                    resolved_fact = fact_reply
        text = resolved_fact or _fallback_preference_reply(user_input)

    lowered = text.strip().lower()
    raw_identity_markers = (
        "i'm an ai assistant",
        "i am an ai assistant",
        "i'm an artificial intelligence model",
        "i am an artificial intelligence model",
        "artificial intelligence model known as llama",
        "large language model meta ai",
        "i am llama",
        "i'm llama",
        "known as llama",
        "released to the public in",
        "released in 2023",
        "i'm not your dad",
        "i'm dadbot, and i'm here with you",
        "i am dadbot, and i am here with you",
        "i don't have a personal life outside this chat",
        "i do not have a personal life outside this chat",
        "i exist solely to provide information",
        "i don't have personal experiences",
        "i do not have personal experiences",
        "i don't have a physical existence",
        "i do not have a physical existence",
        "i don't have a personal preference",
        "i do not have a personal preference",
        "i don't have preferences or feelings",
        "i do not have preferences or feelings",
        "i don't have personal preferences",
        "i do not have personal preferences",
        "i don't have personal experiences",
        "i do not have personal experiences",
        "i'm an ai designed",
        "i am an ai designed",
        "i'm a large language model",
        "i am a large language model",
        "i don't participate in physical activities",
        "i do not participate in physical activities",
    )
    if any(marker in lowered for marker in raw_identity_markers):
        dad_bio = getattr(finalization, "_dad_bio_reply", None)
        if callable(dad_bio):
            with suppress(Exception):
                text = str(dad_bio() or text)
        if not text or any(marker in str(text).lower() for marker in raw_identity_markers):
            if any(marker in user_text for marker in preference_prompt_markers):
                text = _fallback_preference_reply(user_input)
            elif any(marker in user_text for marker in detail_markers):
                text = _fallback_dad_full_story()
            else:
                text = _fallback_dad_story()

    if any(marker in user_text for marker in age_prompt_markers) and any(
        marker in str(text).lower() for marker in raw_identity_markers
    ):
        text = _fallback_dad_age()

    if any(marker in user_text for marker in preference_prompt_markers) and any(
        marker in str(text).lower() for marker in raw_identity_markers
    ):
        text = _fallback_preference_reply(user_input)

    finalize = getattr(finalization, "finalize", None)
    if callable(finalize):
        with suppress(Exception):
            return str(finalize(text, "neutral", user_input=user_input) or text)

    append_signoff = getattr(finalization, "append_signoff", None)
    if callable(append_signoff):
        with suppress(Exception):
            return str(append_signoff(text) or text)
    return text

avatar_path = Path("static/assets/dad_avatar.jpg")
if avatar_path.exists():
    st.sidebar.image(str(avatar_path), width=80)
st.sidebar.markdown(
    """
    <div style="padding:0.25rem 0 0.9rem;">
      <div style="font-size:0.72rem; letter-spacing:0.16em; text-transform:uppercase; opacity:0.78;">Family runtime</div>
      <div style="font-size:1.75rem; font-weight:700; line-height:1; margin-top:0.2rem;">DadBot</div>
    </div>
    """,
    unsafe_allow_html=True,
)
page = st.sidebar.radio("Navigation", ["Chat", "Smart Home", "Voice", "Status", "Workshop"])
_render_thread_sidebar(page)
if page != "Chat":
    _render_app_shell(page)

if page == "Chat":
    st.markdown('<div class="chat-layout">', unsafe_allow_html=True)
    main_col = st.container()
    _install_client_speech_cancel_guard()
    st.sidebar.checkbox(
        "Auto-speak replies",
        key="auto_speak_replies",
        value=bool(st.session_state.get("auto_speak_replies", True)),
        help="When enabled, Dad speaks each assistant reply automatically.",
    )
    auto_speak_replies = _auto_speak_replies_enabled()

    with main_col:
        st.markdown('<div class="main-column">', unsafe_allow_html=True)
        embed_url = _spatial_avatar_embed_url()
        embed_src = _spatial_embed_src(embed_url)
        avatar_markup = _presence_avatar_markup()

        st.markdown(
            f"""
            <section class="presence-shell">
                <div class="presence-pill">DadBot</div>
                {('' if embed_url else f'<div class="presence-avatar-wrap">{avatar_markup}</div>')}
            </section>
            """,
            unsafe_allow_html=True,
        )
        if embed_url:
            components.html(_spatial_avatar_inline_html(embed_src), height=720, scrolling=False)

        for msg in _thread_messages():
            with st.chat_message(msg["role"]):
                st.markdown(msg["text"])
                audio_payload = str(msg.get("audio_base64") or "").strip()
                if audio_payload and auto_speak_replies:
                    with suppress(Exception):
                        st.audio(
                            base64.b64decode(audio_payload),
                            format=str(msg.get("audio_mime") or "audio/mpeg"),
                            autoplay=True,
                        )
                st.caption(msg["time"])

        try:
            chat_input_value = st.chat_input("Message Dad...", accept_file="multiple")
        except TypeError:
            chat_input_value = st.chat_input("Message Dad...")

        prompt, uploaded_files = _chat_input_payload(chat_input_value)
        if prompt or uploaded_files:
            attachments, attachment_issues = media_service.process_upload(uploaded_files)
            for issue in list(attachment_issues or []):
                st.warning(str(issue))
            if not prompt and attachments:
                prompt = "Please review the attached files."

            now = datetime.datetime.now().strftime("%H:%M")
            _append_chat_message("user", prompt, stamp=now)
            _record_runtime_event(
                "chat_user_message",
                "User submitted message",
                payload={"length": len(prompt), "attachments": len(attachments)},
            )
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.spinner("Dad is thinking..."):
                try:
                    reply, _ = bot.process_user_message(prompt, attachments=attachments)
                    reply = _normalize_ui_reply(reply, user_input=prompt)
                    _record_runtime_event("chat_response_ok", "Assistant response generated", payload={"length": len(reply)})
                except Exception as exc:
                    traceback.print_exc()
                    fallback = _local_prompt_fallback(prompt)
                    if fallback:
                        reply = fallback
                        _record_runtime_event(
                            "chat_response_fallback",
                            "Assistant recovered with local fallback",
                            severity="warning",
                            payload={"error": str(exc)},
                        )
                    else:
                        reply = f"Sorry, I hit a snag: {exc}"
                        _record_runtime_event("chat_response_error", "Assistant response failed", severity="error", payload={"error": str(exc)})
                    _record_guardrail("turn_execution_error", str(exc), severity="error")
                    _queue_failed_message(prompt, str(exc))

            assistant_audio_b64 = ""
            assistant_audio_mime = ""
            with st.chat_message("assistant"):
                st.markdown(reply)
                with suppress(Exception):
                    if auto_speak_replies:
                        manager = TTSManager(bot)
                        audio_bytes, _audio_error, audio_mime = manager.get_tts_audio(reply)
                        if audio_bytes:
                            assistant_audio_b64 = base64.b64encode(audio_bytes).decode("ascii")
                            assistant_audio_mime = str(audio_mime or "audio/mpeg")
                            st.audio(audio_bytes, format=assistant_audio_mime, autoplay=True)
                        else:
                            _speak_reply_client_side(reply)
            _append_chat_message(
                "assistant",
                reply,
                stamp=now,
                audio_base64=assistant_audio_b64,
                audio_mime=assistant_audio_mime,
            )
    st.markdown('</div>', unsafe_allow_html=True)

elif page == "Smart Home":
    st.caption("MQTT controls are live when a broker is configured.")

    broker = st.text_input("Broker host", value=str(st.session_state.get("smarthome_broker") or "localhost"))
    port = st.number_input("Broker port", min_value=1, max_value=65535, value=int(st.session_state.get("smarthome_port") or 1883), step=1)
    st.session_state.smarthome_broker = broker
    st.session_state.smarthome_port = int(port)

    topic = st.text_input("Topic", value="dadbot/home/living_room/light")
    payload = st.text_input("Payload", value="ON")

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Connect MQTT", use_container_width=True):
            try:
                from dadbot.smart_home.mqtt_client import SmartHomeMQTTClient

                client = SmartHomeMQTTClient(broker, int(port))
                client.connect()
                st.session_state.mqtt_client = client
                st.success("Connected to MQTT broker")
            except Exception as exc:
                st.error(f"Connect failed: {exc}")
    with col2:
        if st.button("Publish", use_container_width=True):
            client = st.session_state.get("mqtt_client")
            if client is None:
                st.warning("Connect first")
            else:
                try:
                    client.publish(topic, payload)
                    st.success(f"Published to {topic}")
                except Exception as exc:
                    st.error(f"Publish failed: {exc}")
    with col3:
        if st.button("Disconnect", use_container_width=True):
            client = st.session_state.get("mqtt_client")
            if client is not None:
                with suppress(Exception):
                    client.disconnect()
            st.session_state.mqtt_client = None
            st.info("Disconnected")

elif page == "Voice":
    st.caption("Restored voice controls from the previous UI surface.")
    try:
        render_voice_controls(bot)
        render_realtime_voice_call(bot)
    except Exception as exc:
        st.error(f"Voice surface failed: {exc}")

elif page == "Status":
    _render_guardrail_strip()
    shell = dict(bot.ui_shell_snapshot() or {})
    ollama = dict(shell.get("ollama") or {})

    c1, c2, c3 = st.columns(3)
    c1.metric("Active Thread", str(shell.get("active_thread_id") or "default"))
    c2.metric("Mood", str(shell.get("last_mood") or "neutral"))
    c3.metric("Threads", str(len(list(shell.get("threads") or []))))

    st.subheader("Model Runtime")
    st.json(
        {
            "model": str(ollama.get("model") or "llama3.2:latest"),
            "connected": bool(ollama.get("connected", False)),
            "models_count": int(ollama.get("models_count") or 0),
            "error": str(ollama.get("error") or ""),
        },
    )

    if st.button("Probe one turn", use_container_width=True):
        try:
            probe_reply, _ = bot.process_user_message("Health probe: reply with one short sentence.")
            probe_reply = _normalize_ui_reply(probe_reply, user_input="Health probe: reply with one short sentence.")
            _record_runtime_event("status_probe_ok", "Status probe completed")
            st.success(str(probe_reply or "ok"))
        except Exception as exc:
            _record_runtime_event("status_probe_error", "Status probe failed", severity="error", payload={"error": str(exc)})
            _record_guardrail("status_probe_error", str(exc), severity="error")
            st.error(f"Probe failed: {exc}")

    st.subheader("Runtime Timeline")
    _render_runtime_timeline(limit=25)

    st.subheader("Guardrail Feed")
    guardrails = list(st.session_state.get("runtime_guardrails") or [])[-20:]
    if not guardrails:
        st.info("No guardrail events recorded.")
    else:
        st.dataframe(
            [
                {
                    "time": str(item.get("time") or ""),
                    "thread": str(item.get("thread_id") or "default"),
                    "rule": str(item.get("rule") or ""),
                    "severity": str(item.get("severity") or "warning"),
                    "detail": str(item.get("detail") or ""),
                }
                for item in reversed(guardrails)
            ],
            use_container_width=True,
        )

elif page == "Workshop":
    st.caption("Operational tools restored for fast runtime inspection and recovery.")
    _render_guardrail_strip()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Run live probe", use_container_width=True):
            try:
                reply, _ = bot.process_user_message("Workshop probe: confirm system ready.")
                reply = _normalize_ui_reply(reply, user_input="Workshop probe: confirm system ready.")
                _record_runtime_event("workshop_probe_ok", "Workshop probe completed")
                st.success(str(reply or "ok"))
            except Exception as exc:
                _record_runtime_event("workshop_probe_error", "Workshop probe failed", severity="error", payload={"error": str(exc)})
                _record_guardrail("workshop_probe_error", str(exc), severity="error")
                st.error(f"Probe failed: {exc}")

        if st.button("Clear chat history", use_container_width=True):
            _set_thread_messages([], _active_thread_id())
            _record_runtime_event("thread_cleared", "Cleared active thread messages")
            st.info("Chat history cleared")

    with col2:
        if st.button("Reset MQTT session", use_container_width=True):
            client = st.session_state.get("mqtt_client")
            if client is not None:
                with suppress(Exception):
                    client.disconnect()
            st.session_state.mqtt_client = None
            _record_runtime_event("mqtt_reset", "Reset MQTT session")
            st.info("MQTT session reset")

        if st.button("Reset voice runtime state", use_container_width=True):
            st.session_state.voice_runtime_state = {}
            _record_runtime_event("voice_state_reset", "Reset voice runtime state")
            st.info("Voice runtime state reset")

    st.subheader("Outbox")
    queue = list(st.session_state.get("message_outbox") or [])
    if not queue:
        st.info("Outbox is empty.")
    else:
        st.dataframe(
            [
                {
                    "time": str(item.get("time") or ""),
                    "thread": str(item.get("thread_id") or "default"),
                    "prompt": str(item.get("prompt") or "")[:120],
                    "error": str(item.get("error") or "")[:160],
                }
                for item in reversed(queue[-25:])
            ],
            use_container_width=True,
        )

        retry_col, clear_col = st.columns(2)
        with retry_col:
            if st.button("Retry queued", use_container_width=True):
                retried = 0
                remaining = []
                for item in queue:
                    prompt = str(item.get("prompt") or "").strip()
                    thread_id = str(item.get("thread_id") or "default")
                    if not prompt:
                        continue
                    _switch_active_thread(thread_id)
                    try:
                        now = datetime.datetime.now().strftime("%H:%M")
                        _append_chat_message("user", prompt, stamp=now, thread_id=thread_id)
                        reply, _ = bot.process_user_message(prompt)
                        reply = _normalize_ui_reply(reply, user_input=prompt)
                        _append_chat_message("assistant", reply, stamp=now, thread_id=thread_id)
                        retried += 1
                    except Exception as exc:
                        remaining.append(
                            {
                                "thread_id": thread_id,
                                "prompt": prompt,
                                "error": str(exc),
                                "time": _now_stamp(),
                            }
                        )
                        _record_guardrail("outbox_retry_error", str(exc), severity="error")
                st.session_state.message_outbox = remaining
                _record_runtime_event("outbox_retry", f"Retried {retried} queued messages")
                st.success(f"Retried {retried} queued messages")
                st.rerun()
        with clear_col:
            if st.button("Clear outbox", use_container_width=True):
                st.session_state.message_outbox = []
                _record_runtime_event("outbox_cleared", "Cleared message outbox")
                st.info("Outbox cleared")

    st.subheader("Export / Import Active Thread")
    active_payload = {
        "thread_id": _active_thread_id(),
        "messages": _thread_messages(),
    }
    st.download_button(
        "Download active thread JSON",
        data=json.dumps(active_payload, indent=2, ensure_ascii=True),
        file_name=f"{_active_thread_id()}-thread.json",
        mime="application/json",
        use_container_width=True,
    )

    uploaded_thread = st.file_uploader("Import thread JSON", type=["json"], key="workshop-import-thread")
    if uploaded_thread is not None and st.button("Import into new thread", use_container_width=True):
        try:
            parsed = json.loads(uploaded_thread.getvalue().decode("utf-8", errors="replace"))
            incoming = list(parsed.get("messages") or []) if isinstance(parsed, dict) else []
            cleaned = [dict(item) for item in incoming if isinstance(item, dict)]
            new_tid = _create_new_thread()
            _set_thread_messages(cleaned, new_tid)
            _record_runtime_event("thread_imported", f"Imported thread into {new_tid}")
            st.success(f"Imported thread into {new_tid}")
            st.rerun()
        except Exception as exc:
            _record_guardrail("thread_import_error", str(exc), severity="error")
            st.error(f"Import failed: {exc}")