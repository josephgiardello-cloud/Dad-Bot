from __future__ import annotations

import streamlit as st

try:
    from streamlit_webrtc import WebRtcMode, RTCConfiguration as WebRtcRTCConfiguration, webrtc_streamer
    WEBRTC_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    WEBRTC_AVAILABLE = False
    webrtc_streamer = None
    WebRtcMode = None
    WebRtcRTCConfiguration = None


def voice_known_devices(voice: dict, runtime_state: dict) -> list[str]:
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


def persist_known_devices(voice: dict, known_devices: list[str]) -> None:
    cleaned = []
    for item in list(known_devices or []):
        normalized = str(item or "").strip()
        if normalized and normalized not in cleaned:
            cleaned.append(normalized)
    if "default" not in cleaned:
        cleaned.insert(0, "default")
    voice["known_device_ids"] = cleaned[:12]


def collect_webrtc_audio_bytes(webrtc_ctx, *, key: str, min_bytes: int) -> bytes:
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


def render_voice_capture_layer(controller, voice: dict, *, key_prefix: str) -> bytes:
    min_audio_bytes = max(3000, int(voice.get("barge_in_min_audio_bytes") or 3500))
    runtime_state = controller.runtime_state if isinstance(controller.runtime_state, dict) else {}

    known_devices = voice_known_devices(voice, runtime_state)
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
            persist_known_devices(voice, known_devices)
            controller.set_device(custom)
            st.rerun()

    persist_known_devices(voice, known_devices)
    voice["mic_preference"] = str(selected_device or "default")
    controller.set_device(selected_device)

    if not WEBRTC_AVAILABLE:
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

    return collect_webrtc_audio_bytes(webrtc_ctx, key=f"{key_prefix}-webrtc-capture", min_bytes=min_audio_bytes)
