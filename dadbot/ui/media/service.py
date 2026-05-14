from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import uuid

import streamlit as st

from dadbot.ui.helpers import (
    local_stt_backend_status,
    local_tts_backend_status,
    render_voice_dependency_help,
)
from dadbot.ui.prefs_state import voice_preferences
from dadbot.ui.voice_control_plane import VoiceSessionController

from . import stt, tts, uploads, webrtc

logger = logging.getLogger(__name__)

_WEBRTC_EXPERIMENTAL_ENABLED = str(
    os.environ.get("DADBOT_ENABLE_EXPERIMENTAL_WEBRTC_CALL") or "",
).strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


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
    from dadbot.ui import interaction_controller

    api = interaction_controller.get_chat_event_api()
    payload = voice_profile_payload(voice)
    current = api.PROFILE.get("voice", {}) if isinstance(api.PROFILE, dict) else {}
    if not isinstance(current, dict):
        current = {}
    if all(current.get(key) == value for key, value in payload.items()):
        return
    api.update_voice_profile(payload)
    api.save_profile()


def get_voice_session_controller(bot) -> VoiceSessionController:
    from dadbot.ui import interaction_controller

    api = interaction_controller.get_chat_event_api()
    runtime_state = st.session_state.setdefault("voice_runtime_state", {})
    controller = VoiceSessionController(
        voice_preferences(),
        runtime_state=runtime_state,
        ledger_emitter=lambda event_type, payload: interaction_controller.emit_voice_runtime_ledger_event(
            event_type, payload
        ),
        session_id_provider=lambda: str(api.active_thread_id or "voice-ui"),
        trace_id_provider=lambda: st.session_state.setdefault(
            "voice_trace_id",
            uuid.uuid4().hex,
        ),
    )
    if "voice_profile_fingerprint" not in st.session_state:
        st.session_state.voice_profile_fingerprint = hashlib.sha1(
            json.dumps(
                voice_profile_payload(controller.voice_config),
                sort_keys=True,
                default=str,
            ).encode("utf-8"),
        ).hexdigest()
    return controller


class MediaService:
    def process_upload(self, uploaded_files):
        return uploads.build_chat_attachments_from_uploads(uploaded_files)

    def handle_audio_input(self, bot):
        return self.render_voice_controls(bot)

    def synthesize_reply(self, bot, reply_text: str) -> None:
        self.render_reply_tts(bot, reply_text)

    def render_voice_controls(self, bot):
        controller = get_voice_session_controller(bot)
        voice = controller.voice_config
        snapshot = controller.snapshot()

        with st.container(border=True):
            st.subheader("Voice Control Plane")
            st.caption(
                "One voice state machine controls listening, recording, processing, and speaking.",
            )

            mode = st.radio(
                "Voice mode",
                options=["push_to_talk", "always_listening", "ambient"],
                index={"push_to_talk": 0, "always_listening": 1, "ambient": 2}.get(
                    str(snapshot.get("mode") or "push_to_talk"),
                    0,
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
            status_col2.markdown(
                f"**Mic:** {'Ready' if snapshot.get('mic_available') else 'Waiting'}",
            )

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
            voice["interruptions_enabled"] = bool(
                st.session_state.get("voice-interruptions-enabled", True),
            )
            voice["barge_in_enabled"] = bool(
                st.session_state.get("voice-barge-in-enabled", True),
            )
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
                ),
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
                    },
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
            if interrupt_col1.button(
                "Cancel active voice turn",
                key="voice-cancel-active",
                use_container_width=True,
            ):
                controller.request_cancel(reason="manual_cancel", priority="high")
                snapshot = controller.snapshot()
            if interrupt_col2.button(
                "Priority override: high",
                key="voice-priority-high",
                use_container_width=True,
            ):
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
                st.caption(
                    "Ambient mode active: capture runs in the background listener loop.",
                )
                persist_voice_profile_if_changed(bot, voice)
                return None

            audio_bytes = webrtc.render_voice_capture_layer(
                controller,
                voice,
                key_prefix="voice-control",
            )
            if not audio_bytes:
                persist_voice_profile_if_changed(bot, voice)
                return None
            controller.start_turn(
                turn_id=uuid.uuid4().hex,
                priority=str(snapshot.get("priority_override") or "normal"),
            )
            clip_hash = hashlib.sha1(audio_bytes).hexdigest() if audio_bytes else ""
            transcript_key = f"voice-transcript:{clip_hash}"
            transcript_text = str(st.session_state.get(transcript_key) or "")
            transcript_error = ""

            if not transcript_text:
                with st.spinner("Transcribing locally..."):
                    transcript_text, transcript_error = controller.process_audio_capture(
                        audio_bytes,
                        transcribe_fn=lambda payload: stt.transcribe_audio_bytes(
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

            if mode == "always_listening" and bool(
                voice.get("auto_send_always_listening", True),
            ):
                st.info("Always-listening captured and queued this utterance.")
                persist_voice_profile_if_changed(bot, voice)
                return edited

            if st.button(
                "Send transcript",
                key=f"voice-send:{clip_hash}",
                type="primary",
            ):
                persist_voice_profile_if_changed(bot, voice)
                return edited
            persist_voice_profile_if_changed(bot, voice)
            return None

    def render_ambient_voice_listener(self, bot):
        controller = get_voice_session_controller(bot)
        voice = controller.voice_config
        if not bool(voice.get("enabled")) or str(voice.get("mode") or "") != "ambient":
            return

        st.markdown(
            "<div style='display:flex;align-items:center;gap:0.6rem;'>"
            "<span style='width:10px;height:10px;border-radius:50%;background:#22c55e;'"
            "animation:pulse 1.5s infinite;display:inline-block;'></span>"
            "<span style='font-size:0.85rem;opacity:0.8;'>Ambient listener active - speak naturally</span>"
            "</div>",
            unsafe_allow_html=True,
        )

        stt_ready, stt_backend, stt_message = local_stt_backend_status(voice)
        if not stt_ready:
            controller.mark_error(stt_message)
            st.caption(f"STT offline: {stt_message}")
            return
        controller.mark_mic_available(True)

        audio_bytes = webrtc.render_voice_capture_layer(
            controller,
            voice,
            key_prefix="ambient-voice",
        )
        if not audio_bytes or len(audio_bytes) < 4000:
            return

        controller.start_turn(turn_id=uuid.uuid4().hex, priority="normal")

        clip_hash = hashlib.sha1(audio_bytes).hexdigest()
        processed_key = f"ambient-processed:{clip_hash}"
        if st.session_state.get(processed_key):
            return

        st.session_state[processed_key] = True

        transcript_text, transcript_error = controller.process_audio_capture(
            audio_bytes,
            transcribe_fn=lambda payload: stt.transcribe_audio_bytes(
                payload,
                backend=stt_backend,
                model_name=str(voice.get("stt_model") or "base"),
                language=str(voice.get("stt_language") or "en"),
            ),
        )
        if transcript_error or not transcript_text:
            persist_voice_profile_if_changed(bot, voice)
            return

        queue = st.session_state.setdefault("ambient_utterance_queue", [])
        queue.append(transcript_text.strip())
        st.toast(f'Dad heard: "{transcript_text[:60]}"', icon="ðŸŽ™ï¸")
        persist_voice_profile_if_changed(bot, voice)

    def render_realtime_voice_call(self, bot):
        from dadbot.ui import interaction_controller, state_manager

        controller = get_voice_session_controller(bot)
        voice = controller.voice_config

        if not _WEBRTC_EXPERIMENTAL_ENABLED:
            st.info(
                "Real-time WebRTC calling is currently disabled for stability. "
                "Use the Voice input panel in Chat (push-to-talk) for reliable local STT/TTS. "
                "Set DADBOT_ENABLE_EXPERIMENTAL_WEBRTC_CALL=1 to re-enable this experimental mode.",
            )
            return

        if not webrtc.WEBRTC_AVAILABLE:
            st.info(
                "Install `streamlit-webrtc` to enable real-time voice calls:\n```\npip install streamlit-webrtc\n```",
            )
            return

        if not voice.get("enabled", False):
            st.info(
                "Enable voice in Preferences -> Voice to use the real-time call feature.",
            )
            return

        st.subheader("ðŸ“ž Talk to Dad - Live")
        st.caption(
            "Hands-free, real-time voice conversation. Uses your mic -> STT -> Dad -> TTS pipeline.",
        )

        known_devices = webrtc.voice_known_devices(
            voice,
            controller.runtime_state if isinstance(controller.runtime_state, dict) else {},
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
        webrtc.persist_known_devices(voice, known_devices)

        rtc_config = webrtc.WebRtcRTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        )

        media_audio: dict | bool = True
        if str(selected_device or "default") != "default":
            media_audio = {"deviceId": {"exact": str(selected_device)}}

        webrtc_ctx = webrtc.webrtc_streamer(
            key="dadbot-voice-call",
            mode=webrtc.WebRtcMode.SENDRECV,
            rtc_configuration=rtc_config,
            media_stream_constraints={"video": False, "audio": media_audio},
            async_processing=True,
        )

        if not (webrtc_ctx and webrtc_ctx.state.playing):
            st.caption(
                "Click **START** above, allow microphone access, then speak naturally.",
            )
            return

        st.success("Live call connected - speak naturally!")

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
                if len(audio_bytes) > 8000:
                    stt_ready, stt_backend, _ = local_stt_backend_status(voice)
                    if stt_ready:
                        transcript, _err = controller.process_audio_capture(
                            audio_bytes,
                            transcribe_fn=lambda payload: stt.transcribe_audio_bytes(
                                payload,
                                backend=stt_backend,
                                model_name=str(voice.get("stt_model") or "base"),
                                language=str(voice.get("stt_language") or "en"),
                            ),
                            min_audio_bytes=8000,
                        )
                        if transcript and transcript.strip():
                            st.session_state["realtime_transcript"] = transcript.strip()

        live_transcript = str(st.session_state.get("realtime_transcript") or "").strip()
        if live_transcript:
            st.markdown(f"**You said:** {live_transcript}")

        send_col, clear_col = st.columns([3, 1])
        if send_col.button(
            "Send to Dad",
            type="primary",
            key="webrtc-send-btn",
            use_container_width=True,
        ):
            prompt = str(st.session_state.get("realtime_transcript") or "").strip()
            if prompt:
                with st.spinner("Dad is thinking..."):
                    try:
                        runtime_result = interaction_controller.process_prompt_via_runtime(
                            thread_id=str(
                                interaction_controller.get_chat_event_api().active_thread_id or "default",
                            ),
                            prompt=prompt,
                            attachments=[],
                        )
                        reply_text = str(runtime_result.get("reply") or "")

                        if reply_text:
                            st.markdown(f"**Dad:** {reply_text}")
                            self.render_reply_tts(bot, reply_text)
                    except Exception as exc:
                        state_manager.record_runtime_rejection(
                            exc,
                            action="realtime_voice_send",
                        )
                        st.error(f"Dad Runtime blocked or failed this request: {exc}")

                st.session_state["realtime_transcript"] = ""
            else:
                st.warning("Nothing transcribed yet - speak first, then click Send.")
        if clear_col.button("Clear", key="webrtc-clear-btn", use_container_width=True):
            st.session_state["realtime_transcript"] = ""
            st.rerun()

        st.caption(
            "Tip: speak naturally, wait a moment for transcription, then press Send.",
        )

    def render_reply_tts(self, bot, reply_text):
        controller = get_voice_session_controller(bot)
        voice = controller.voice_config
        if (
            not bool(voice.get("enabled"))
            or not bool(voice.get("tts_enabled", True))
            or bool(voice.get("muted", False))
        ):
            return

        cancel_state = controller.consume_cancel()
        if bool(cancel_state.get("cancel_requested")) and bool(
            voice.get("allow_tts_cancel", True),
        ):
            controller.complete_turn()
            return

        text = str(reply_text or "").strip()
        if not text:
            return

        cache = st.session_state.setdefault("voice_tts_cache", {})
        key = hashlib.sha1(
            (
                text + "|" + str(voice.get("tts_voice") or "warm_dad") + "|" + str(int(voice.get("tts_rate") or 0))
            ).encode("utf-8"),
        ).hexdigest()

        audio_bytes = cache.get(key)
        error = ""
        if audio_bytes is None:
            tts_backend = str(voice.get("tts_backend") or "pyttsx3").strip().lower()
            piper_model = str(voice.get("tts_piper_model_path") or "").strip()
            controller.begin_speaking()
            with st.spinner("Generating local Dad voice audio..."):
                if tts_backend == "piper" or (tts_backend == "auto" and shutil.which("piper") and piper_model):
                    audio_bytes, error = tts.synthesize_piper_audio(
                        text,
                        model_path=piper_model,
                    )
                else:
                    audio_bytes, error = tts.synthesize_tts_audio(
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
            if bool(pending_cancel.get("cancel_requested")) and bool(
                voice.get("allow_tts_cancel", True),
            ):
                controller.complete_turn()
                persist_voice_profile_if_changed(bot, voice)
                return
            st.audio(
                audio_bytes,
                format="audio/wav",
                autoplay=bool(voice.get("tts_autoplay", False)),
            )
            controller.complete_turn()
        persist_voice_profile_if_changed(bot, voice)


media_service = MediaService()
