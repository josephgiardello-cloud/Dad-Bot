import asyncio
import base64
import json
import logging
import datetime
import os
import traceback
import uuid
from collections.abc import Callable, Coroutine
from contextlib import suppress
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import streamlit as st
import streamlit.components.v1 as components

from dadbot.components.voice import render_realtime_voice_call, render_voice_controls
from dadbot.config import DadBotConfig, DadRuntimeConfig
from dadbot.contracts import DadBotContext
from dadbot.core.dadbot import DadBot
from dadbot.core.runtime_errors import NON_FATAL_RUNTIME_EXCEPTIONS
from dadbot.managers.tts import TTSManager
from dadbot.runtime.mcp.approval_service import MCPApprovalService
from dadbot.runtime.mcp.tool_client import (
    MCPToolClient,
    configured_ha_mcp_tools,
    configured_ha_mcp_url,
)
from dadbot.ui.media.service import MediaService
from dadbot.ui.source_grounding import render_source_mode
from dadbot.ui.streamlit_context import (
    SQLiteStreamlitThreadStore,
    StreamlitBotPool,
    StreamlitContextService,
    StreamlitReflectionService,
    serialize_thread_messages,
    set_runtime_thread_id,
    streamlit_turn_lock,
)
from dadbot_system.contracts import DEFAULT_MODEL


def _now_stamp() -> str:
    return datetime.datetime.now().strftime("%H:%M:%S")


def _restore_thread_selection(threads: dict[str, list[dict[str, Any]]]) -> None:
    requested = st.query_params.get("thread")
    active = st.session_state.get("active_thread_id", "default")
    if requested is not None:
        if requested in threads:
            active = requested
        else:
            st.warning("The requested conversation is unavailable for this tenant. Showing your current conversation.")
    if active not in threads:
        active = "default"
    st.session_state.active_thread_id = active
    if st.session_state.get("thread-picker") != active:
        st.session_state["thread-picker"] = active


def _ensure_ui_runtime_state() -> None:
    tenant_id = st.session_state.bot.config.tenant_id
    previous_tenant = st.session_state.get("_streamlit_ui_tenant")
    if previous_tenant is not None and previous_tenant != tenant_id:
        for key in (
            "chat_history", "chat_threads", "active_thread_id", "thread-picker",
            "runtime_events", "runtime_guardrails", "message_outbox",
        ):
            st.session_state.pop(key, None)
    st.session_state["_streamlit_ui_tenant"] = tenant_id
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "chat_threads" not in st.session_state:
        seeded = list(st.session_state.chat_history or [])
        thread_store = getattr(st.session_state.get("bot"), "_streamlit_thread_store", None)
        load_threads = getattr(thread_store, "load_threads", None)
        restored = load_threads() if callable(load_threads) else {}
        st.session_state.chat_threads = restored or {"default": seeded}
    threads = st.session_state["chat_threads"]
    if not isinstance(threads, dict):
        raise TypeError("Streamlit conversation state must be a thread mapping")
    threads.setdefault("default", [])
    st.session_state.chat_threads = threads

    _restore_thread_selection(threads)

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

        div[data-testid="stButton"] button,
        div[data-testid="stDownloadButton"] button {
            border-radius: 14px;
            border: 1px solid rgba(39, 75, 95, 0.45);
            background: linear-gradient(180deg, rgba(255,255,255,0.96), rgba(244,240,233,0.92));
            color: #142338 !important;
            font-weight: 600;
            box-shadow: 0 10px 24px rgba(22, 33, 47, 0.08);
            transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease;
        }

        div[data-testid="stButton"] button *,
        div[data-testid="stDownloadButton"] button *,
        [data-testid="stSidebar"] div[data-testid="stButton"] button *,
        [data-testid="stSidebar"] div[data-testid="stDownloadButton"] button * {
            color: #142338 !important;
        }

        div[data-testid="stButton"] button:hover,
        div[data-testid="stDownloadButton"] button:hover {
            transform: translateY(-1px);
            border-color: rgba(39, 75, 95, 0.72);
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
        "Smart Home": ("Home control", "Home Assistant MCP controls with an explicit tool allowlist."),
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
    thread_store = getattr(st.session_state.get("bot"), "_streamlit_thread_store", None)
    save_thread = getattr(thread_store, "save_thread", None)
    if callable(save_thread):
        save_thread(tid, threads[tid])
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
        raise ValueError("The requested conversation is unavailable for this tenant")
    active_bot = st.session_state.get("bot")
    if active_bot is not None:
        with streamlit_turn_lock(active_bot):
            set_runtime_thread_id(active_bot, tid)
            active_bot._streamlit_active_thread_id = tid
    st.session_state.active_thread_id = tid
    st.session_state.chat_history = _thread_messages(tid)
    if st.query_params.get("thread") != tid:
        st.query_params["thread"] = tid


def _select_active_thread() -> None:
    _switch_active_thread(st.session_state["thread-picker"])


def _create_new_thread() -> str:
    tid = f"thread-{uuid.uuid4().hex[:8]}"
    _set_thread_messages([], tid)
    _switch_active_thread(tid)
    st.session_state["thread-picker"] = tid
    _record_runtime_event("thread_created", f"Created {tid}")
    return tid


def _fork_active_thread() -> str:
    source_id = _active_thread_id()
    source = _thread_messages(source_id)
    tid = f"fork-{uuid.uuid4().hex[:8]}"
    _set_thread_messages(source, tid)
    _switch_active_thread(tid)
    st.session_state["thread-picker"] = tid
    _record_runtime_event("thread_forked", f"Forked into {tid}")
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


def _queue_failed_message(prompt: str, error: str, *, source_mode: bool = False) -> None:
    queue = list(st.session_state.get("message_outbox") or [])
    queue.append(
        {
            "thread_id": _active_thread_id(),
            "prompt": str(prompt or ""),
            "error": str(error or ""),
            "time": _now_stamp(),
            "source_mode": source_mode,
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
    st.sidebar.selectbox(
        "Active conversation",
        thread_ids,
        index=None if "thread-picker" in st.session_state else index,
        key="thread-picker",
        on_change=_select_active_thread,
    )

    col1, col2 = st.columns(2) if chat_mode else st.sidebar.columns(2)
    with col1:
        st.button(
            "New chat" if chat_mode else "New",
            use_container_width=True,
            on_click=_create_new_thread,
        )
    with col2:
        st.button(
            "Duplicate" if chat_mode else "Fork",
            use_container_width=True,
            on_click=_fork_active_thread,
        )

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


def _build_bot(tenant_id: str) -> DadBot:
    runtime_config = DadRuntimeConfig()
    config = DadBotConfig(runtime_config=runtime_config, tenant_id=tenant_id)
    model_name = config.llm_model

    from dadbot.registry import ServiceRegistry

    bot = DadBot(
        model_name=model_name,
        config=config,
        runtime_config=runtime_config,
    )

    canonical_service_container = bot.services
    registry = ServiceRegistry.boot(
        config_path="config.yaml", bot=bot, registry=canonical_service_container.registry,
    )
    context_service = StreamlitContextService(bot, context_service=registry.get("context_service"))
    registry.register("context_service", context_service)
    registry.register("memory", context_service)
    mcp_service = MCPApprovalService(bot)
    registry.register("mcp_tool_service", mcp_service)
    canonical_persistence = registry.get("persistence_service")
    internal_state_manager = bot.internal_state_manager
    if canonical_persistence is None:
        raise RuntimeError("The Streamlit runtime requires the canonical PersistenceService")
    canonical_service_container.persistence_service = canonical_persistence
    prompt_assembly = getattr(bot, "prompt_assembly", None)
    if not callable(getattr(prompt_assembly, "build_request_system_prompt", None)):
        from dadbot.managers.prompt_assembly import PromptAssemblyManager

        bot.prompt_assembly = PromptAssemblyManager(DadBotContext.from_runtime(bot))
    registry.register("reflection", StreamlitReflectionService(bot, internal_state_manager))

    setattr(bot, "_streamlit_thread_store", SQLiteStreamlitThreadStore(tenant_id=config.tenant_id))

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


@st.cache_resource(show_spinner=False, on_release=StreamlitBotPool.close)
def _runtime_pool() -> StreamlitBotPool:
    return StreamlitBotPool()


_STREAMLIT_RUNTIME_BUILD = "2026-11-11-action-review-ui-v24"

st.set_page_config(
    page_title="DadBot",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "bot" not in st.session_state or st.session_state.get("_streamlit_runtime_build") != _STREAMLIT_RUNTIME_BUILD:
    previous_bot = st.session_state.get("bot")
    if previous_bot is not None and not getattr(previous_bot, "_streamlit_managed_shared", False):
        previous_bot.shutdown()
    tenant_id = DadBotConfig().tenant_id
    st.session_state.bot = _runtime_pool().get(tenant_id, _STREAMLIT_RUNTIME_BUILD, _build_bot)
    st.session_state._streamlit_runtime_build = _STREAMLIT_RUNTIME_BUILD

_ensure_ui_runtime_state()
_switch_active_thread(_active_thread_id())
_inject_modern_theme()

bot = st.session_state.bot
media_service = MediaService()


def _process_user_message_with_thread(
    user_input: str,
    attachments: list[dict[str, Any]] | None = None,
    *,
    source_mode: bool = False,
):
    with streamlit_turn_lock(bot):
        thread_id = _active_thread_id()
        set_runtime_thread_id(bot, thread_id)
        bot._streamlit_active_thread_id = thread_id
        bot._streamlit_context_messages = serialize_thread_messages(_thread_messages(), user_input)
        try:
            if source_mode:
                return bot.process_user_message(user_input, attachments=attachments, source_mode=True)
            return bot.process_user_message(user_input, attachments=attachments)
        finally:
            bot._streamlit_context_messages = []


def _resolve_pending_mcp_approval(service: Any, *, approved: bool, request_id: str) -> bool:
    return _run_mcp_continuation(
        lambda thread_id: service.resolve_mcp_approval(thread_id, approved=approved, request_id=request_id),
        user_text=service.decision_text(approved=approved),
        failure_message=(
            "The approval turn could not finish. An action may already have run; "
            "it will not be automatically resent. Check its audit status before trying a new action."
        ),
    )


def _run_mcp_continuation(
    continuation: Callable[[str], Coroutine[Any, Any, str]], *, user_text: str, failure_message: str,
) -> bool:
    with streamlit_turn_lock(bot):
        thread_id = _active_thread_id()
        set_runtime_thread_id(bot, thread_id)
        bot._streamlit_active_thread_id = thread_id
        bot._streamlit_context_messages = list(_thread_messages())
        try:
            if dict(st.session_state.get("thread_source_modes") or {}).get(thread_id):
                raise ValueError("MCP continuations cannot run in source mode")
            reply = asyncio.run(continuation(thread_id))
        except NON_FATAL_RUNTIME_EXCEPTIONS:
            logging.getLogger(__name__).exception("Canonical MCP continuation failed")
            st.error(failure_message)
            return False
        finally:
            bot._streamlit_context_messages = []
        _append_chat_message("user", user_text, stamp=_now_stamp())
        _append_chat_message("assistant", bot.reply_finalization.normalize_reply(reply), stamp=_now_stamp())
        return True


def _render_mcp_action_reviews() -> None:
    st.subheader("Home Assistant action review")
    service = bot.services.registry.get("mcp_tool_service", optional=True)
    if not callable(getattr(service, "get_mcp_action_history", None)):
        st.info("Action review is unavailable: the MCP approval service is not wired.")
        return
    with streamlit_turn_lock(bot):
        thread_id = _active_thread_id()
        set_runtime_thread_id(bot, thread_id)
        bot._streamlit_active_thread_id = thread_id
        try:
            actions = service.get_mcp_action_history(thread_id)
        except NON_FATAL_RUNTIME_EXCEPTIONS:
            logging.getLogger(__name__).exception("Durable MCP action history could not be read")
            st.error("Could not read action history. Review is blocked until the ledger is recovered and verified.")
            return
    if not actions:
        st.info("No recorded Home Assistant actions in this conversation.")
        return
    st.dataframe(
        [{
            "approval": action.call.request_id, "tool": action.call.tool_name,
            "recorded_outcome": action.status, "user_confirmation": action.review_outcome or "not recorded",
            "retry_policy": action.retry_policy,
        } for action in reversed(actions[-25:])],
        use_container_width=True,
    )
    unresolved = [action for action in actions if action.status == "uncertain" and not action.review_outcome]
    st.caption(
        "This is a ledger-backed review queue, not an automatic retry queue. "
        "User confirmations do not certify tool success or authorize replay. "
        "Any repeat action requires a new proposal and approval."
    )
    if not unresolved:
        st.info("No uncertain actions awaiting your review.")
        return
    by_id = {action.call.request_id: action for action in unresolved}
    selected_id = st.selectbox(
        "Uncertain action to review", list(by_id),
        format_func=lambda request_id: f"{by_id[request_id].call.tool_name} | {request_id}",
        key=f"action-review-selection-{thread_id}",
    )
    action = by_id[selected_id]
    st.code(
        f"Approval: {action.call.request_id}\nOriginal trace: {action.call.trace_id}\n"
        f"Approved input hash: {action.call.input_hash}\nOutcome event: {action.outcome_event_id}",
        language=None,
    )
    st.warning(
        "Check the original proposal and Home Assistant's action history. "
        "A device's current state alone does not prove whether this action happened. "
        "If you cannot identify or verify the action, leave it unresolved."
    )
    source_mode = bool(dict(st.session_state.get("thread_source_modes") or {}).get(thread_id))
    if source_mode:
        st.info("Switch off source mode in Chat before recording an action review.")
    checked = st.checkbox(
        "I checked this exact action in Home Assistant",
        key=f"action-reviewed-{thread_id}-{action.outcome_event_id}", disabled=source_mode,
    )
    performed_col, not_performed_col = st.columns(2)
    performed = performed_col.button("I verified the action happened", disabled=source_mode or not checked)
    not_performed = not_performed_col.button("I verified no action happened", disabled=source_mode or not checked)
    if not (performed or not_performed):
        return
    outcome = "performed" if performed else "not_performed"
    text = service.review_text(outcome=outcome, request_id=action.call.request_id)
    if _run_mcp_continuation(
        lambda active_thread: service.review_mcp_action(
            active_thread, request_id=action.call.request_id, outcome_event_id=action.outcome_event_id, outcome=outcome,
        ),
        user_text=text,
        failure_message=(
            "The action-review turn could not finish. Your confirmation may already be recorded; "
            "reload and check action history before retrying. This review does not send an external action."
        ),
    ):
        st.rerun()


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
    source_mode = render_source_mode(st, bot, _active_thread_id())
    st.markdown('<div class="chat-layout">', unsafe_allow_html=True)
    main_col = st.container()
    _install_client_speech_cancel_guard()
    st.sidebar.checkbox(
        "Auto-speak replies",
        key="auto_speak_replies",
        value=bool(st.session_state.get("auto_speak_replies", True)),
        disabled=source_mode,
        help="When enabled, Dad speaks each assistant reply automatically.",
    )
    auto_speak_replies = _auto_speak_replies_enabled() and not source_mode
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

        registry = getattr(getattr(bot, "services", None), "registry", None)
        mcp_service = registry.get("mcp_tool_service", optional=True) if registry is not None else None
        pending_approval = (
            mcp_service.get_pending_mcp_approval(_active_thread_id())
            if mcp_service is not None and callable(getattr(mcp_service, "get_pending_mcp_approval", None))
            else None
        )
        if pending_approval is not None:
            st.warning("Review this exact Home Assistant action. It will not run unless you approve it.")
            if source_mode:
                st.info("Switch off source mode to approve or reject this pending action.")
            st.write("Tool:", pending_approval.tool_name)
            st.code(json.dumps(pending_approval.arguments, ensure_ascii=False, indent=2), language="json")
            approve_col, reject_col = st.columns(2)
            if approve_col.button(
                "Approve this action", key=f"mcp-approve-{pending_approval.request_id}", disabled=source_mode,
            ):
                with st.spinner("Revalidating and executing the approved action..."):
                    completed = _resolve_pending_mcp_approval(
                        mcp_service, approved=True, request_id=pending_approval.request_id,
                    )
                if completed:
                    st.rerun()
            if reject_col.button(
                "Reject this action", key=f"mcp-reject-{pending_approval.request_id}", disabled=source_mode,
            ):
                if _resolve_pending_mcp_approval(mcp_service, approved=False, request_id=pending_approval.request_id):
                    st.rerun()
            chat_input_value = None
        elif source_mode:
            chat_input_value = st.chat_input("Ask Jor-El about approved sources...")
        else:
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

            assistant_message = st.chat_message("assistant")
            with assistant_message:
                assistant_output = st.empty()

            with st.spinner("Dad is thinking..."):
                try:
                    reply, _ = _process_user_message_with_thread(
                        prompt,
                        attachments=attachments,
                        source_mode=source_mode,
                    )
                    reply = bot.reply_finalization.normalize_reply(reply)
                    _record_runtime_event("chat_response_ok", "Assistant response generated", payload={"length": len(reply)})
                except Exception as exc:
                    traceback.print_exc()
                    reply = f"Sorry, I hit a snag: {exc}"
                    _record_runtime_event("chat_response_error", "Assistant response failed", severity="error", payload={"error": str(exc)})
                    _record_guardrail("turn_execution_error", str(exc), severity="error")
                    _queue_failed_message(prompt, str(exc), source_mode=source_mode)

            assistant_audio_b64 = ""
            assistant_audio_mime = ""
            reply_thread_id = _active_thread_id()
            _append_chat_message("assistant", reply, stamp=now, thread_id=reply_thread_id)
            reply_message_index = len(_thread_messages(reply_thread_id)) - 1
            if (
                mcp_service is not None and callable(getattr(mcp_service, "get_pending_mcp_approval", None))
                and mcp_service.get_pending_mcp_approval(reply_thread_id) is not None
            ):
                st.rerun()
            with assistant_message:
                assistant_output.markdown(reply)
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
            if assistant_audio_b64:
                messages = _thread_messages(reply_thread_id)
                messages[reply_message_index]["audio_base64"] = assistant_audio_b64
                messages[reply_message_index]["audio_mime"] = assistant_audio_mime
                _set_thread_messages(messages, reply_thread_id)
    st.markdown('</div>', unsafe_allow_html=True)

elif page == "Smart Home":
    st.caption("Home Assistant actions are provided by an explicitly configured MCP server.")
    mcp_url = configured_ha_mcp_url()
    allowed_tools = configured_ha_mcp_tools()

    if not mcp_url:
        st.warning("Home Assistant MCP is not configured.")
        st.code("DADBOT_HA_MCP_URL\nDADBOT_HA_MCP_TOKEN\nDADBOT_HA_MCP_ALLOWED_TOOLS", language="text")
    else:
        st.write({"MCP endpoint": mcp_url, "Allowed tool names": sorted(allowed_tools)})
        if not allowed_tools:
            st.warning("No Home Assistant tools are allowlisted; chat will not execute home actions.")
        elif st.button("Check Home Assistant MCP", use_container_width=True):
            async def _check_mcp_connection() -> list[str]:
                client = MCPToolClient(
                    mcp_url,
                    bearer_token=str(os.environ.get("DADBOT_HA_MCP_TOKEN") or ""),
                )
                async with client.connect() as session:
                    tools = await client.list_allowed_tools(session, allowed_tools)
                    return [str(tool.get("name") or "") for tool in tools]

            try:
                available_tools = asyncio.run(_check_mcp_connection())
                if available_tools:
                    st.success(f"Connected. {len(available_tools)} allowlisted tool(s) are available.")
                    st.write(available_tools)
                else:
                    st.warning("Connected, but none of the configured tool names were offered by the MCP server.")
            except Exception as exc:
                st.error(f"MCP connection failed: {exc}")

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
            "model": str(ollama.get("model") or DEFAULT_MODEL),
            "connected": bool(ollama.get("connected", False)),
            "models_count": int(ollama.get("models_count") or 0),
            "error": str(ollama.get("error") or ""),
        },
    )

    if st.button("Probe one turn", use_container_width=True):
        try:
            probe_reply, _ = _process_user_message_with_thread("Health probe: reply with one short sentence.")
            probe_reply = bot.reply_finalization.normalize_reply(probe_reply)
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
    _render_mcp_action_reviews()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Run live probe", use_container_width=True):
            try:
                reply, _ = _process_user_message_with_thread("Workshop probe: confirm system ready.")
                reply = bot.reply_finalization.normalize_reply(reply)
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
                    source_mode = item.get("source_mode") is True
                    if not prompt:
                        continue
                    _switch_active_thread(thread_id)
                    try:
                        now = datetime.datetime.now().strftime("%H:%M")
                        _append_chat_message("user", prompt, stamp=now, thread_id=thread_id)
                        reply, _ = _process_user_message_with_thread(prompt, source_mode=source_mode)
                        reply = bot.reply_finalization.normalize_reply(reply)
                        _append_chat_message("assistant", reply, stamp=now, thread_id=thread_id)
                        retried += 1
                    except Exception as exc:
                        remaining.append(
                            {
                                "thread_id": thread_id,
                                "prompt": prompt,
                                "error": str(exc),
                                "time": _now_stamp(),
                                "source_mode": source_mode,
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